"""Calibrated INT4 quantization (AWQ, GPTQ): the algorithms reduce output error versus
round-to-nearest, AWQ's scale folding preserves the network function, and calibrated exports are
read correctly by the C++ reference model."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch

from redline import calibrate as K
from redline import convert as C
from redline.format import RedlineFile
from redline.testing import hf_logits, make_tiny_hf_model

TOKENS = [1, 17, 42, 99, 7, 300, 12, 5, 64, 128, 256, 3]


def run_ref(ref_bin: str, model_path: str, tokens: list[int], tmp_path) -> np.ndarray:
    out = tmp_path / "logits.f32"
    subprocess.run([ref_bin, "--model", model_path, "--tokens", ",".join(map(str, tokens)), "--logits-out",
                    str(out)], check=True)
    return np.fromfile(out, dtype=np.float32).reshape(len(tokens), -1)


def outlier_inputs(t: int, k: int, seed: int) -> torch.Tensor:
    """Activations with a few large, correlated channels, like real LLM hidden states."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(t, 8, generator=g) @ torch.randn(8, k, generator=g) * 0.3
    x = base + torch.randn(t, k, generator=g)
    x[:, torch.randperm(k, generator=g)[: k // 32]] *= 20.0
    return x


def out_err(w, wq, x) -> float:
    return float(((x @ (w - wq).t()) ** 2).mean())


def test_gptq_reduces_output_error_vs_rtn():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(96, 256, generator=g) * 0.05
    x = outlier_inputs(1024, 256, 2)
    s, o = K.group_params(w, 64)
    rtn = K.dequant(K.quantize_given(w, s, o, 64), s, o, 64)
    q, s2, o2 = K.gptq_quantize(w, x.t() @ x, 64)
    gptq = K.dequant(q, s2, o2, 64)
    assert q.dtype == torch.uint8 and int(q.max()) <= 15
    assert out_err(w, gptq, x) < 0.7 * out_err(w, rtn, x)


def test_awq_search_never_worse_than_rtn_and_helps_outliers():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(128, 256, generator=g) * 0.05
    x = outlier_inputs(512, 256, 4)
    s, alpha = K.awq_search(w, x, x.abs().mean(0), 64)
    sc, of = K.group_params(w * s, 64, ratios=(1.0,))
    awq = K.dequant(K.quantize_given(w * s, sc, of, 64), sc, of, 64) / s
    sc0, of0 = K.group_params(w, 64, ratios=(1.0,))
    rtn = K.dequant(K.quantize_given(w, sc0, of0, 64), sc0, of0, 64)
    assert alpha > 0 and out_err(w, awq, x) < out_err(w, rtn, x)


def test_pack_layout_matches_quantizer():
    q = torch.randint(0, 16, (4, 8), dtype=torch.uint8)
    p = K.pack_int4(q)
    assert torch.equal(p & 0xF, q[:, 0::2]) and torch.equal(p >> 4, q[:, 1::2])


@pytest.mark.parametrize("method", ["awq", "gptq", "awq_gptq"])
def test_calibrated_export_matches_dequantized_hf(ref_bin, tmp_path, method):
    hf_dir = str(tmp_path / "hf")
    fp_model = make_tiny_hf_model(hf_dir, seed=5, num_heads=4, num_kv_heads=4 if method == "awq" else 2)
    out = str(tmp_path / f"{method}.rlm")
    C.convert(hf_dir, out, weight_dtype="int4", lm_head_dtype="int8", group_size=64, device="cpu",
              verbose=False, method=method, calib_samples=8, calib_seqlen=32, calib_dataset="random")
    rf = RedlineFile(out)
    assert rf.spec.weight_dtype == "int4"
    # The dequantized checkpoint (with AWQ-modified norms) is exactly what the engine computes.
    import transformers

    deq = transformers.AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32)
    missing, unexpected = deq.load_state_dict(rf.to_hf_state_dict(), strict=False)
    assert not unexpected and all("rotary" in m for m in missing)
    expected = hf_logits(deq.eval(), TOKENS).numpy()
    ref = run_ref(ref_bin, out, TOKENS, tmp_path)
    assert float(np.abs(ref - expected).max() / (np.abs(expected).max() + 1e-12)) < 1e-4
    # Loose sanity bound against the unquantized model (random weights amplify INT4 error).
    fp = hf_logits(fp_model, TOKENS).numpy()
    cos = (ref * fp).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(fp, axis=-1))
    # (GPTQ fits its Hessian from only 256 random calibration tokens here, so it can overfit.)
    assert cos.min() > 0.7, cos


def test_awq_folding_preserves_function(tmp_path):
    """Scaling W by s and folding 1/s into the preceding op is exact before quantization."""
    hf_dir = str(tmp_path / "hf")
    model = make_tiny_hf_model(hf_dir, seed=6, num_heads=4, num_kv_heads=4).eval()
    ref = hf_logits(model, TOKENS)
    layer = model.model.layers[0]
    h = layer.self_attn.q_proj.weight.shape[1]
    s = torch.rand(h) + 0.5
    for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
        lin.weight.data *= s
    layer.input_layernorm.weight.data /= s
    inter = layer.mlp.down_proj.weight.shape[1]
    s2 = torch.rand(inter) + 0.5
    layer.mlp.down_proj.weight.data *= s2
    layer.mlp.up_proj.weight.data /= s2[:, None]
    assert torch.allclose(hf_logits(model, TOKENS), ref, atol=1e-4, rtol=1e-4)
