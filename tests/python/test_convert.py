"""End-to-end exporter tests: HF checkpoint -> .rlm -> C++ reference model.

The C++ CPU reference (`redline_ref`) must reproduce Hugging Face logits for
FP16 exports, and the dequantized-HF logits for INT8/INT4 exports. This
validates tensor naming/fusion, RoPE convention, GQA, biases, tied
embeddings, rope scaling and the quantized storage layout.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess

import numpy as np
import pytest
import torch

from redline import convert as C
from redline import quantize as Q
from redline.format import RedlineFile, weight_bytes
from redline.testing import hf_logits, make_tiny_hf_model

TOKENS = [1, 17, 300, 42, 9, 77, 5, 255, 3, 128, 64, 500]


def run_ref(ref_bin: str, model_path: str, tokens: list[int], tmp_path) -> np.ndarray:
    out = tmp_path / "logits.f32"
    subprocess.run(
        [ref_bin, "--model", model_path, "--tokens", ",".join(map(str, tokens)), "--logits-out", str(out)],
        check=True,
    )
    return np.fromfile(out, dtype=np.float32).reshape(len(tokens), -1)


def hf_model_from_state_dict(hf_dir: str, state_dict: dict):
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, unexpected
    assert all("rotary" in m for m in missing), missing
    return model.eval()


def max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


@pytest.mark.parametrize(
    "arch,tie,rope_scaling",
    [
        ("llama", False, None),
        ("llama", True, None),
        ("qwen2", False, None),
        (
            "llama",
            False,
            {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
             "original_max_position_embeddings": 64},
        ),
    ],
)
def test_fp16_export_matches_hf(ref_bin, tmp_path, arch, tie, rope_scaling):
    hf_dir = str(tmp_path / "hf")
    model = make_tiny_hf_model(hf_dir, seed=1, arch=arch, tie_word_embeddings=tie, rope_scaling=rope_scaling)
    out = str(tmp_path / "m.rlm")
    spec = C.convert(hf_dir, out, weight_dtype="fp16", verbose=False)
    assert spec.tie_word_embeddings == tie
    assert spec.attention_bias == (arch == "qwen2")
    if rope_scaling:
        assert spec.rope_scaling["type"] == "llama3"
    ref = run_ref(ref_bin, out, TOKENS, tmp_path)
    expected = hf_logits(model, TOKENS).numpy()
    assert ref.shape == expected.shape
    assert max_rel_err(ref, expected) < 1e-4


@pytest.mark.parametrize("qtype,group", [("int8", 128), ("int4", 64), ("int4", 128)])
def test_quantized_export_matches_dequantized_hf(ref_bin, tmp_path, qtype, group):
    hf_dir = str(tmp_path / "hf")
    fp_model = make_tiny_hf_model(hf_dir, seed=2)
    out = str(tmp_path / "q.rlm")
    C.convert(hf_dir, out, weight_dtype=qtype, lm_head_dtype="int8", group_size=group, verbose=False)
    rf = RedlineFile(out)
    assert rf.spec.weight_dtype == qtype and rf.spec.group_size == group
    deq_model = hf_model_from_state_dict(hf_dir, rf.to_hf_state_dict())
    ref = run_ref(ref_bin, out, TOKENS, tmp_path)
    expected = hf_logits(deq_model, TOKENS).numpy()
    assert max_rel_err(ref, expected) < 1e-4

    # Sanity bound only: random Gaussian weights have no outlier structure and
    # amplify errors, so this is far looser than what real checkpoints show
    # (use redline.eval_quality for real quality measurements).
    fp = hf_logits(fp_model, TOKENS).numpy()
    cos = (ref * fp).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(fp, axis=-1))
    assert cos.min() > (0.995 if qtype == "int8" else 0.8), cos


def test_streaming_writer_produces_valid_safetensors(tmp_path):
    from safetensors import safe_open

    hf_dir = str(tmp_path / "hf")
    make_tiny_hf_model(hf_dir, seed=3)
    out = str(tmp_path / "m.rlm")
    C.convert(hf_dir, out, weight_dtype="int4", group_size=64, verbose=False)
    with safe_open(out, framework="pt") as f:
        meta = f.metadata()
        assert meta["format"] == "redline"
        cfg = json.loads(meta["redline_config"])
        assert cfg["quant"]["weight_dtype"] == "int4"
        w = f.get_tensor("layers.0.wqkv.weight")
        assert w.dtype == torch.uint8 and tuple(w.shape) == (256 + 2 * 128, 128)
        assert f.get_tensor("layers.1.w_down.scales").shape == (256, 512 // 64)
    with open(out, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        assert n % 8 == 0
    assert weight_bytes(out) == os.path.getsize(out) - 8 - n


def test_quantizers_roundtrip_bounds():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.05
    q, s = Q.quantize_int8(w, clip_search=False)
    deq = Q.dequantize_int8(q, s)
    assert (deq - w).abs().max() <= 0.5 * s.float().max() + 1e-6
    p, s4, o4 = Q.quantize_int4(w, group_size=64, clip_search=False)
    deq4 = Q.dequantize_int4(p, s4, o4, 64)
    assert (deq4 - w).abs().max() <= 0.5 * s4.float().max() + 1e-3
    # Clip search never increases the reconstruction error.
    p2, s2, o2 = Q.quantize_int4(w, group_size=64, clip_search=True)
    err_rtn = ((deq4 - w) ** 2).sum()
    err_clip = ((Q.dequantize_int4(p2, s2, o2, 64) - w) ** 2).sum()
    assert err_clip <= err_rtn + 1e-6


def test_cpp_and_python_int4_layout_agree(ref_bin, tmp_path):
    """A model written by the C++ generator reads back correctly in Python."""
    out = str(tmp_path / "cpp.rlm")
    subprocess.run([ref_bin, "--write-random-model", out, "--quant", "int4", "--seed", "5"], check=True)
    rf = RedlineFile(out)
    w = rf.linear("layers.0.wqkv", "int4")
    assert w.shape == (rf.spec.qkv_size, rf.spec.hidden_size)
    assert torch.isfinite(w).all()
