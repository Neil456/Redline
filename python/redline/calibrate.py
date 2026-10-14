"""Calibrated INT4 quantization: AWQ (activation-aware scaling) and GPTQ (error compensation).

Plain round-to-nearest treats every weight as equally important. Real activations are not uniform:
a few input channels carry much larger values, and errors on the weights they multiply dominate
the layer's output error. Both methods use a small calibration set to account for that:

* AWQ (Lin et al., 2023) scales each input channel of a linear layer by s = mean|x|^alpha before
  quantizing (W' = W diag(s)) and folds 1/s into the op that produces the input (an RMSNorm
  weight, or the up projection for down_proj), so the network computes the same function but the
  salient channels get finer quantization steps. alpha is grid-searched per layer to minimize
  the output error on calibration activations.
* GPTQ (Frantar et al., 2022) quantizes one input column at a time and spreads each column's
  rounding error onto the columns not yet quantized, weighted by the inverse Hessian
  H = X^T X of the layer inputs: a second-order (OBS) update that minimizes ||(W - Q) X||^2.

`awq_gptq` applies the AWQ scales first and then quantizes with GPTQ. Group scales and offsets
(w = q * s + o, 4-bit, groups along K) keep Redline's INT4 format, so calibrated models run on the
same kernels. Layers are processed one at a time: the calibration batch is pushed through layer
i with its already-quantized weights to produce the inputs of layer i + 1 (as GPTQ does).

Used by `python -m redline.convert --weights int4 --method awq|gptq|awq_gptq`.
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field

import torch

from .quantize import INT4_CLIP_RATIOS, _MIN_SCALE, _fp16_round

METHODS = ("rtn", "awq", "gptq", "awq_gptq")


# --------------------------------------------------------------------------
# Calibration data
# --------------------------------------------------------------------------


def calibration_tokens(tokenizer, dataset: str, n_samples: int, seqlen: int, seed: int = 0,
                       vocab_size: int | None = None) -> torch.Tensor:
    """[n_samples, seqlen] token windows. C4 is the usual choice (and keeps WikiText for eval)."""
    rng = random.Random(seed)
    if dataset == "random":
        hi = vocab_size or len(tokenizer)
        g = torch.Generator().manual_seed(seed)
        return torch.randint(min(100, hi - 1), hi, (n_samples, seqlen), generator=g)
    from datasets import load_dataset

    if dataset == "c4":
        ds = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                          split="validation")
        texts = ds["text"]
    elif dataset == "wikitext":
        texts = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip()]
        texts = ["\n".join(texts)]
    else:
        raise ValueError(f"unknown calibration dataset {dataset}")
    samples = []
    if len(texts) == 1:  # one long document: random windows
        ids = tokenizer(texts[0], add_special_tokens=False)["input_ids"]
        for _ in range(n_samples):
            i = rng.randrange(0, len(ids) - seqlen)
            samples.append(ids[i : i + seqlen])
    else:  # GPTQ-style: a random window of a random document long enough
        tries = 0
        while len(samples) < n_samples:
            tries += 1
            if tries > 100 * n_samples:
                raise RuntimeError("not enough long calibration documents")
            ids = tokenizer(texts[rng.randrange(len(texts))], add_special_tokens=False)["input_ids"]
            if len(ids) <= seqlen:
                continue
            i = rng.randrange(0, len(ids) - seqlen)
            samples.append(ids[i : i + seqlen])
    return torch.tensor(samples, dtype=torch.long)


# --------------------------------------------------------------------------
# Quantization primitives (Redline INT4: q in [0, 15], w = q * s + o, fp16 s / o)
# --------------------------------------------------------------------------


def group_params(w: torch.Tensor, group_size: int, col_weight: torch.Tensor | None = None,
                 ratios=INT4_CLIP_RATIOS, chunk_rows: int = 2048):
    """Per-group (scale, offset) for w [N, K], choosing the clip ratio that minimizes the squared
    error, optionally weighted per input column (activation energy). Returns fp32 tensors holding
    fp16-representable values, each [N, K / group_size]."""
    n, k = w.shape
    groups = k // group_size
    s_out = torch.empty((n, groups), dtype=torch.float32, device=w.device)
    o_out = torch.empty_like(s_out)
    cw = None if col_weight is None else col_weight.float().reshape(1, groups, group_size)
    for r0 in range(0, n, chunk_rows):
        x = w[r0 : r0 + chunk_rows].float().reshape(-1, groups, group_size)
        lo, hi = x.amin(dim=2), x.amax(dim=2)
        best_err = torch.full_like(lo, float("inf"))
        best_s, best_o = torch.ones_like(lo), torch.zeros_like(lo)
        for ratio in ratios:
            s = _fp16_round(torch.clamp((hi - lo) * ratio / 15.0, min=_MIN_SCALE))
            o = _fp16_round(lo * ratio)
            q = torch.clamp(torch.round((x - o[..., None]) / s[..., None]), 0, 15)
            e = (q * s[..., None] + o[..., None] - x) ** 2
            if cw is not None:
                e = e * cw
            err = e.sum(dim=2)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, s, best_s)
            best_o = torch.where(better, o, best_o)
        flat = hi <= lo
        best_s = torch.where(flat, torch.ones_like(best_s), best_s)
        best_o = torch.where(flat, _fp16_round(lo), best_o)
        s_out[r0 : r0 + chunk_rows] = best_s
        o_out[r0 : r0 + chunk_rows] = best_o
    return s_out, o_out


def quantize_given(w: torch.Tensor, s: torch.Tensor, o: torch.Tensor, group_size: int) -> torch.Tensor:
    """q (uint8 [N, K]) for given group params."""
    n, k = w.shape
    se = s.repeat_interleave(group_size, dim=1)
    oe = o.repeat_interleave(group_size, dim=1)
    return torch.clamp(torch.round((w.float() - oe) / se), 0, 15).to(torch.uint8)


def dequant(q: torch.Tensor, s: torch.Tensor, o: torch.Tensor, group_size: int) -> torch.Tensor:
    return q.float() * s.repeat_interleave(group_size, dim=1) + o.repeat_interleave(group_size, dim=1)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """uint8 [N, K] (0..15) -> [N, K / 2], even column in the low nibble (Redline's layout)."""
    return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()


def gptq_quantize(w: torch.Tensor, h: torch.Tensor, group_size: int, blocksize: int = 128,
                  percdamp: float = 0.01):
    """GPTQ with static groups along K. w [N, K], h = sum x x^T [K, K] (both fp32).

    Returns (q uint8 [N, K], s [N, G], o [N, G]). Group parameters are chosen when the first
    column of a group is reached, from the error-updated weights of that group, with the clip
    search weighted by the Hessian diagonal.
    """
    w = w.clone().float()
    n, k = w.shape
    assert k % group_size == 0
    blocksize = max(blocksize, group_size)
    blocksize -= blocksize % group_size
    h = h.clone().float()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(h))
    h += damp * torch.eye(k, device=h.device)
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    hinv = torch.linalg.cholesky(hinv, upper=True)
    diag_h = torch.diag(h)

    groups = k // group_size
    q_all = torch.zeros((n, k), dtype=torch.uint8, device=w.device)
    s_all = torch.zeros((n, groups), dtype=torch.float32, device=w.device)
    o_all = torch.zeros_like(s_all)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        s = o = None
        for i in range(i2 - i1):
            col = i1 + i
            if col % group_size == 0:
                g = col // group_size
                s_g, o_g = group_params(w1[:, i : i + group_size], group_size,
                                        diag_h[col : col + group_size])
                s, o = s_g[:, 0], o_g[:, 0]
                s_all[:, g], o_all[:, g] = s, o
            wc = w1[:, i]
            qc = torch.clamp(torch.round((wc - o) / s), 0, 15)
            q_all[:, col] = qc.to(torch.uint8)
            e = (wc - (qc * s + o)) / hinv1[i, i]
            w1[:, i:] -= e.unsqueeze(1) @ hinv1[i, i:].unsqueeze(0)
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q_all, s_all, o_all


# --------------------------------------------------------------------------
# Activation statistics
# --------------------------------------------------------------------------


@dataclass
class SiteStats:
    """Statistics of one linear input (shared by the linears that read it)."""

    need_h: bool
    sample_tokens: int
    abs_sum: torch.Tensor | None = None
    sq_sum: torch.Tensor | None = None
    h: torch.Tensor | None = None
    count: int = 0
    samples: list = field(default_factory=list)
    sampled: int = 0

    def add(self, x: torch.Tensor, take: int) -> None:
        x = x.reshape(-1, x.shape[-1])
        xf = x.float()
        if self.abs_sum is None:
            self.abs_sum = torch.zeros(x.shape[-1], device=x.device)
            self.sq_sum = torch.zeros(x.shape[-1], device=x.device)
            if self.need_h:
                self.h = torch.zeros((x.shape[-1], x.shape[-1]), device=x.device)
        self.abs_sum += xf.abs().sum(0)
        self.sq_sum += (xf * xf).sum(0)
        if self.need_h:
            self.h += xf.t() @ xf
        self.count += x.shape[0]
        take = min(take, self.sample_tokens - self.sampled, x.shape[0])
        if take > 0:
            idx = torch.randperm(x.shape[0], device=x.device)[:take]
            self.samples.append(x[idx].clone())
            self.sampled += take

    def scale_inputs(self, s: torch.Tensor) -> None:
        """Inputs become x / s after AWQ folds 1/s into the producing op."""
        self.abs_sum /= s
        self.sq_sum /= s * s
        if self.h is not None:
            self.h /= s[:, None] * s[None, :]
        self.samples = [x / s.to(x.dtype) for x in self.samples]

    @property
    def sample(self) -> torch.Tensor:
        return torch.cat(self.samples, 0)


# --------------------------------------------------------------------------
# AWQ
# --------------------------------------------------------------------------


@torch.no_grad()
def awq_search(w: torch.Tensor, x: torch.Tensor, abs_mean: torch.Tensor, group_size: int, grid: int = 20):
    """Best per-input-channel scale s = mean|x|^alpha for weight w [N, K] given input sample x [T, K]."""
    xf = x.float()
    ref = xf @ w.t()
    best_err, best_s, best_alpha = float("inf"), None, 0.0
    for i in range(grid):
        alpha = i / grid
        s = abs_mean.clamp(min=1e-4).pow(alpha)
        s = s / (s.max() * s.min()).sqrt()
        ws = w * s
        sc, of = group_params(ws, group_size, ratios=(1.0,))
        wq = dequant(quantize_given(ws, sc, of, group_size), sc, of, group_size) / s
        err = ((xf @ wq.t()) - ref).pow(2).mean().item()
        if err < best_err:
            best_err, best_s, best_alpha = err, s, alpha
    return best_s, best_alpha


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _layers(model):
    return model.model.layers


def _capture_layer0_inputs(model, batches, device):
    layers = _layers(model)
    inps, cache = [], {}

    class Catcher(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, hidden_states, *args, **kwargs):
            inps.append(hidden_states)
            cache["kwargs"] = kwargs
            raise ValueError("captured")

    layers[0] = Catcher(layers[0])
    for b in batches:
        try:
            model(b.to(device), use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].m
    kwargs = {k: v for k, v in cache["kwargs"].items() if k not in ("past_key_values", "use_cache")}
    return inps, kwargs


def _run_layer(layer, inp, kwargs):
    out = layer(inp, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def calibrate(model_dir: str, method: str, group_size: int, n_samples: int = 128, seqlen: int = 512,
              dataset: str = "c4", device: str = "cuda", batch_size: int = 8, awq_tokens: int = 1024,
              seed: int = 0, verbose: bool = True) -> dict:
    """Quantizes every decoder-layer linear of the HF model in `model_dir` with `method`.

    Returns {"norms": {hf_name: fp16 tensor}, "linears": {(layer, kind): [packed, scales, offsets]}}
    where kind is one of wqkv / wo / w_gate_up / w_down (Redline's fused layout).
    """
    assert method in ("awq", "gptq", "awq_gptq"), method
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_awq, use_gptq = method in ("awq", "awq_gptq"), method in ("gptq", "awq_gptq")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    t_start = time.time()
    tok = None if dataset == "random" else AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    cfg = model.config
    tokens = calibration_tokens(tok, dataset, n_samples, seqlen, seed, cfg.vocab_size)
    batches = list(tokens.split(batch_size))
    inps, kwargs = _capture_layer0_inputs(model, batches, device)
    take = max(1, math.ceil(awq_tokens / len(batches)))
    mha = cfg.num_attention_heads == (getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
    if verbose:
        print(f"calibrating {method}: {tokens.shape[0]} x {tokens.shape[1]} tokens ({dataset}), "
              f"group {group_size}", file=sys.stderr)

    norms, linears = {}, {}
    for li, layer in enumerate(_layers(model)):
        t0 = time.time()
        attn, mlp = layer.self_attn, layer.mlp
        sites = {name: SiteStats(use_gptq, awq_tokens) for name in ("attn", "o", "ffn", "down")}
        hooks = [
            attn.q_proj.register_forward_pre_hook(lambda m, a: sites["attn"].add(a[0], take)),
            attn.o_proj.register_forward_pre_hook(lambda m, a: sites["o"].add(a[0], take)),
            mlp.gate_proj.register_forward_pre_hook(lambda m, a: sites["ffn"].add(a[0], take)),
            mlp.down_proj.register_forward_pre_hook(lambda m, a: sites["down"].add(a[0], take)),
        ]
        for inp in inps:
            _run_layer(layer, inp, kwargs)
        for h in hooks:
            h.remove()

        groups = {
            "wqkv": (["q_proj", "k_proj", "v_proj"], attn, "attn"),
            "wo": (["o_proj"], attn, "o"),
            "w_gate_up": (["gate_proj", "up_proj"], mlp, "ffn"),
            "w_down": (["down_proj"], mlp, "down"),
        }

        def weight(kind):
            names, mod, _ = groups[kind]
            return torch.cat([getattr(mod, n).weight.float() for n in names], 0)

        alphas = {}
        if use_awq:
            for kind, prev in (("wqkv", layer.input_layernorm), ("w_gate_up", layer.post_attention_layernorm),
                               ("w_down", None), ("wo", None)):
                if kind == "wo" and not mha:
                    continue  # GQA: v_proj and o_proj channels do not correspond 1:1
                st = sites[groups[kind][2]]
                s, alphas[kind] = awq_search(weight(kind), st.sample, st.abs_sum / st.count, group_size)
                names, mod, _ = groups[kind]
                for n in names:  # W' = W diag(s)
                    lin = getattr(mod, n)
                    lin.weight.data = (lin.weight.float() * s).to(lin.weight.dtype)
                if prev is not None:  # fold 1/s into the RMSNorm weight
                    prev.weight.data = (prev.weight.float() / s).to(prev.weight.dtype)
                elif kind == "w_down":  # ... or into the up projection's output rows
                    up = mlp.up_proj
                    up.weight.data = (up.weight.float() / s[:, None]).to(up.weight.dtype)
                    if up.bias is not None:
                        up.bias.data = (up.bias.float() / s).to(up.bias.dtype)
                else:  # wo (MHA): into v_proj's output rows
                    v = attn.v_proj
                    v.weight.data = (v.weight.float() / s[:, None]).to(v.weight.dtype)
                    if v.bias is not None:
                        v.bias.data = (v.bias.float() / s).to(v.bias.dtype)
                st.scale_inputs(s)

        for kind, (names, mod, site) in groups.items():
            st = sites[site]
            w = weight(kind)
            if use_gptq:
                q, s, o = gptq_quantize(w, st.h, group_size)
            else:
                s, o = group_params(w, group_size, col_weight=st.sq_sum / st.count)
                q = quantize_given(w, s, o, group_size)
            wq = dequant(q, s, o, group_size)
            r0 = 0
            for n in names:  # quantized weights feed the next layer's calibration inputs
                lin = getattr(mod, n)
                rows = lin.weight.shape[0]
                lin.weight.data = wq[r0 : r0 + rows].to(lin.weight.dtype)
                r0 += rows
            linears[(li, kind)] = [pack_int4(q).cpu(), s.to(torch.float16).cpu(), o.to(torch.float16).cpu()]

        p = f"model.layers.{li}."
        norms[p + "input_layernorm.weight"] = layer.input_layernorm.weight.detach().to(torch.float16).cpu()
        norms[p + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.detach().to(torch.float16).cpu()
        inps = [_run_layer(layer, inp, kwargs) for inp in inps]
        del sites
        if verbose:
            a = " ".join(f"{k}={v:.2f}" for k, v in alphas.items())
            print(f"  layer {li + 1}/{len(_layers(model))} ({time.time() - t0:.1f}s){' alpha ' + a if a else ''}",
                  file=sys.stderr)
    if verbose:
        print(f"calibration done in {time.time() - t_start:.0f}s", file=sys.stderr)
    del model, inps
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"norms": norms, "linears": linears}
