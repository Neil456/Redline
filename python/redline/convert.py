"""Convert a Hugging Face Llama-family checkpoint into a Redline model file.

Supported: LlamaForCausalLM (Llama 2/3/3.1/3.2, TinyLlama), MistralForCausalLM,
Qwen2ForCausalLM (q/k/v bias). Weights are read shard by shard and written
as a stream, so peak host memory stays around one layer.

    python -m redline.convert --model meta-llama/Llama-3.1-8B-Instruct \\
        --out llama31-8b-int4.rlm --weights int4 --group-size 128 --lm-head int8

INT4 can be calibrated with --method awq | gptq | awq_gptq (see redline.calibrate); the
default (rtn) is round-to-nearest with a per-group clipping search.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

from .format import ModelSpec, linear_plan, quantize_linear, write_streaming, TensorPlan

SUPPORTED_ARCHS = {"LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM"}


# --------------------------------------------------------------------------
# Checkpoint access
# --------------------------------------------------------------------------


def resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"]
    )


class CheckpointReader:
    """Lazily reads tensors from (possibly sharded) safetensors or .bin files."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._handles: dict[str, object] = {}
        self._where: dict[str, str] = {}
        self._bin_state: dict[str, torch.Tensor] | None = None
        st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if st_files:
            from safetensors import safe_open

            for path in st_files:
                h = safe_open(path, framework="pt")
                self._handles[path] = h
                for k in h.keys():
                    self._where[k] = path
        else:
            bins = sorted(glob.glob(os.path.join(model_dir, "pytorch_model*.bin")))
            if not bins:
                raise FileNotFoundError(f"no safetensors or pytorch_model*.bin files in {model_dir}")
            self._bin_state = {}
            for path in bins:
                self._bin_state.update(torch.load(path, map_location="cpu", mmap=True, weights_only=True))

    def has(self, name: str) -> bool:
        if self._bin_state is not None:
            return name in self._bin_state
        return name in self._where

    def get(self, name: str) -> torch.Tensor:
        if self._bin_state is not None:
            return self._bin_state[name]
        return self._handles[self._where[name]].get_tensor(name)


# --------------------------------------------------------------------------
# Config translation
# --------------------------------------------------------------------------


def _rope_settings(cfg: dict) -> tuple[float, dict | None]:
    params = cfg.get("rope_parameters") or {}
    scaling = cfg.get("rope_scaling") or params or None
    theta = cfg.get("rope_theta", params.get("rope_theta", 10000.0))
    if not scaling:
        return float(theta), None
    rtype = scaling.get("rope_type", scaling.get("type", "default"))
    if rtype in ("default", None):
        return float(theta), None
    if rtype == "linear":
        return float(theta), {"type": "linear", "factor": float(scaling["factor"])}
    if rtype == "llama3":
        return float(theta), {
            "type": "llama3",
            "factor": float(scaling["factor"]),
            "low_freq_factor": float(scaling["low_freq_factor"]),
            "high_freq_factor": float(scaling["high_freq_factor"]),
            "original_max_position_embeddings": int(scaling["original_max_position_embeddings"]),
        }
    raise NotImplementedError(f"rope scaling type '{rtype}' is not supported")


def build_spec(model_dir: str, weight_dtype: str, lm_head_dtype: str, group_size: int, source: str) -> ModelSpec:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    archs = set(cfg.get("architectures") or [])
    if not archs & SUPPORTED_ARCHS:
        raise NotImplementedError(f"unsupported architecture {archs}; supported: {sorted(SUPPORTED_ARCHS)}")
    if cfg.get("hidden_act", "silu") != "silu":
        raise NotImplementedError("only SiLU-gated MLPs are supported")
    theta, scaling = _rope_settings(cfg)
    num_heads = cfg["num_attention_heads"]
    eos = cfg.get("eos_token_id")
    gen_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            eos = json.load(f).get("eos_token_id", eos)
    if eos is None:
        eos = []
    elif isinstance(eos, int):
        eos = [eos]
    is_qwen2 = "Qwen2ForCausalLM" in archs
    spec = ModelSpec(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_layers=cfg["num_hidden_layers"],
        num_heads=num_heads,
        num_kv_heads=cfg.get("num_key_value_heads") or num_heads,
        head_dim=cfg.get("head_dim") or cfg["hidden_size"] // num_heads,
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_theta=theta,
        rope_scaling=scaling,
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
        attention_bias=bool(cfg.get("attention_bias", False)) or is_qwen2,
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        bos_token_id=cfg.get("bos_token_id") if cfg.get("bos_token_id") is not None else -1,
        eos_token_ids=list(eos),
        weight_dtype=weight_dtype,
        lm_head_dtype=lm_head_dtype,
        group_size=group_size,
        source=source,
    )
    spec.validate()
    return spec


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def _fused(reader: CheckpointReader, names: list[str]) -> tuple[torch.Tensor, torch.Tensor | None]:
    w = torch.cat([reader.get(n + ".weight").to(torch.float32) for n in names], dim=0)
    biases = []
    any_bias = any(reader.has(n + ".bias") for n in names)
    if any_bias:
        for n in names:
            if reader.has(n + ".bias"):
                biases.append(reader.get(n + ".bias").to(torch.float32))
            else:
                biases.append(torch.zeros(reader.get(n + ".weight").shape[0]))
    return w, (torch.cat(biases) if any_bias else None)


def convert(
    model: str,
    out: str,
    weight_dtype: str = "fp16",
    lm_head_dtype: str = "fp16",
    group_size: int = 128,
    clip_search: bool = True,
    device: str = "cpu",
    verbose: bool = True,
    method: str = "rtn",
    calib_samples: int = 128,
    calib_seqlen: int = 512,
    calib_dataset: str = "c4",
) -> ModelSpec:
    model_dir = resolve_model_dir(model)
    spec = build_spec(model_dir, weight_dtype, lm_head_dtype, group_size, source=model)
    reader = CheckpointReader(model_dir)
    H, I = spec.hidden_size, spec.intermediate_size

    # Every layer must have the same set of biases; probe layer 0.
    def has_bias(names):
        return any(reader.has(f"model.layers.0.{n}.bias") for n in names)

    qkv_names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    gu_names = ["mlp.gate_proj", "mlp.up_proj"]
    bias = {
        "wqkv": has_bias(qkv_names),
        "wo": has_bias(["self_attn.o_proj"]),
        "w_gate_up": has_bias(gu_names),
        "w_down": has_bias(["mlp.down_proj"]),
    }
    spec.attention_bias = bias["wqkv"]

    calib = None
    if method != "rtn":
        if weight_dtype != "int4":
            raise ValueError("--method awq/gptq applies to --weights int4")
        if bias["w_gate_up"] or bias["wo"]:
            raise NotImplementedError("calibrated INT4 does not support MLP / o_proj biases")
        from .calibrate import calibrate

        calib = calibrate(model_dir, method, group_size, n_samples=calib_samples, seqlen=calib_seqlen,
                          dataset=calib_dataset, device=device, verbose=verbose)

    tied = spec.tie_word_embeddings or not reader.has("lm_head.weight")
    spec.tie_word_embeddings = tied
    # A tied FP16 head reuses tok_embeddings; a quantized head is stored separately.
    write_lm_head = (not tied) or lm_head_dtype != "fp16"

    q = weight_dtype
    plan: list[TensorPlan] = [TensorPlan("tok_embeddings.weight", "F16", (spec.vocab_size, H))]
    for i in range(spec.num_layers):
        p = f"layers.{i}."
        plan.append(TensorPlan(p + "attn_norm.weight", "F16", (H,)))
        plan += linear_plan(p + "wqkv", spec.qkv_size, H, q, group_size, bias["wqkv"])
        plan += linear_plan(p + "wo", H, spec.q_size, q, group_size, bias["wo"])
        plan.append(TensorPlan(p + "ffn_norm.weight", "F16", (H,)))
        plan += linear_plan(p + "w_gate_up", 2 * I, H, q, group_size, bias["w_gate_up"])
        plan += linear_plan(p + "w_down", H, I, q, group_size, bias["w_down"])
    plan.append(TensorPlan("norm.weight", "F16", (H,)))
    if write_lm_head:
        plan += linear_plan("lm_head", spec.vocab_size, H, lm_head_dtype, group_size, False)

    def f16(name: str) -> torch.Tensor:
        if calib is not None and name in calib["norms"]:  # AWQ folds 1/s into RMSNorm weights
            return calib["norms"][name]
        return reader.get(name).to(torch.float16)

    def layer_linear(i: int, kind: str, names: list[str]):
        if calib is not None:
            out = list(calib["linears"][(i, kind)])
            _, b = _fused(reader, names) if kind == "wqkv" else (None, None)
            if b is not None:
                out.append(b.to(torch.float16))
            return out
        w, b = _fused(reader, names)
        return linear(w, b, q)

    def linear(w: torch.Tensor, b: torch.Tensor | None, qtype: str):
        tensors = quantize_linear(w.to(device), qtype, group_size, clip_search)
        out = [t.cpu() for t in tensors]
        if b is not None:
            out.append(b.to(torch.float16))
        return out

    def produce():
        yield f16("model.embed_tokens.weight")
        for i in range(spec.num_layers):
            t0 = time.time()
            hp = f"model.layers.{i}."
            yield f16(hp + "input_layernorm.weight")
            yield from layer_linear(i, "wqkv", [hp + n for n in qkv_names])
            yield from layer_linear(i, "wo", [hp + "self_attn.o_proj"])
            yield f16(hp + "post_attention_layernorm.weight")
            yield from layer_linear(i, "w_gate_up", [hp + n for n in gu_names])
            yield from layer_linear(i, "w_down", [hp + "mlp.down_proj"])
            if verbose:
                print(f"  layer {i + 1}/{spec.num_layers} ({time.time() - t0:.1f}s)", file=sys.stderr)
        yield f16("model.norm.weight")
        if write_lm_head:
            src = "model.embed_tokens.weight" if tied else "lm_head.weight"
            yield from linear(reader.get(src).to(torch.float32), None, lm_head_dtype)

    metadata = {"format": "redline", "redline_config": json.dumps(spec.to_json()), "quant_method": method}
    if verbose:
        print(f"converting {model} -> {out} (weights={q}, method={method}, lm_head={lm_head_dtype})",
              file=sys.stderr)
    write_streaming(out, plan, metadata, produce())
    if verbose:
        total = sum(t.nbytes for t in plan)
        print(f"wrote {out}: {len(plan)} tensors, {total / 2**30:.2f} GiB", file=sys.stderr)
    return spec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF hub id or local checkpoint directory")
    ap.add_argument("--out", required=True, help="output .rlm path")
    ap.add_argument("--weights", default="fp16", choices=["fp16", "int8", "int4"])
    ap.add_argument("--lm-head", default="fp16", choices=["fp16", "int8", "int4"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--no-clip-search", action="store_true", help="plain round-to-nearest")
    ap.add_argument("--method", default="rtn", choices=["rtn", "awq", "gptq", "awq_gptq"],
                    help="INT4 quantization: round-to-nearest, or calibrated AWQ / GPTQ / both")
    ap.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext", "random"])
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="device used for quantization math")
    args = ap.parse_args(argv)
    convert(
        args.model,
        args.out,
        weight_dtype=args.weights,
        lm_head_dtype=args.lm_head,
        group_size=args.group_size,
        clip_search=not args.no_clip_search,
        device=args.device,
        method=args.method,
        calib_samples=args.calib_samples,
        calib_seqlen=args.calib_seqlen,
        calib_dataset=args.calib_dataset,
    )


if __name__ == "__main__":
    main()
