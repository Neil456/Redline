"""The Redline model file format (.rlm).

A Redline model is a single safetensors file. ``__metadata__`` holds
``format = "redline"`` and ``redline_config`` (JSON, see ``ModelSpec``).
Tensor names (N = output features, K = input features):

    tok_embeddings.weight            F16 [vocab, hidden]
    layers.{i}.attn_norm.weight      F16 [hidden]
    layers.{i}.wqkv.*                fused q/k/v projection   N = (Hq + 2 Hkv) * D
    layers.{i}.wo.*                  output projection
    layers.{i}.ffn_norm.weight       F16 [hidden]
    layers.{i}.w_gate_up.*           fused gate/up projection N = 2 * intermediate
    layers.{i}.w_down.*              down projection
    norm.weight                      F16 [hidden]
    lm_head.*                        absent when tied to tok_embeddings and FP16

A linear layer ``<p>`` is stored as one of
    fp16:  <p>.weight F16 [N, K]
    int8:  <p>.weight I8 [N, K],  <p>.scales F16 [N]
    int4:  <p>.weight U8 [N, K/2], <p>.scales F16 [N, K/g], <p>.offsets F16 [N, K/g]
plus an optional <p>.bias F16 [N].

The writer streams tensors to disk one at a time so an 8B model can be
converted without holding a second copy of the weights in memory.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from . import quantize as Q

QUANT_TYPES = ("fp16", "int8", "int4")

_ST_DTYPES = {
    torch.float16: "F16",
    torch.float32: "F32",
    torch.bfloat16: "BF16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.int32: "I32",
    torch.int64: "I64",
}
_ITEMSIZE = {"F16": 2, "F32": 4, "BF16": 2, "I8": 1, "U8": 1, "I32": 4, "I64": 8}


@dataclass
class ModelSpec:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 4096
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    bos_token_id: int = -1
    eos_token_ids: list = field(default_factory=list)
    weight_dtype: str = "fp16"
    lm_head_dtype: str = "fp16"
    group_size: int = 128
    source: str = ""

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_size(self) -> int:
        return self.q_size + 2 * self.kv_size

    def to_json(self) -> dict:
        return {
            "architecture": "llama",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "max_position_embeddings": self.max_position_embeddings,
            "attention_bias": self.attention_bias,
            "tie_word_embeddings": self.tie_word_embeddings,
            "bos_token_id": self.bos_token_id,
            "eos_token_ids": list(self.eos_token_ids),
            "quant": {
                "weight_dtype": self.weight_dtype,
                "lm_head_dtype": self.lm_head_dtype,
                "group_size": self.group_size,
            },
            "source": self.source,
        }

    @staticmethod
    def from_json(d: dict) -> "ModelSpec":
        q = d.get("quant", {})
        return ModelSpec(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_layers"],
            num_heads=d["num_heads"],
            num_kv_heads=d.get("num_kv_heads", d["num_heads"]),
            head_dim=d.get("head_dim", d["hidden_size"] // d["num_heads"]),
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=d.get("rope_theta", 10000.0),
            rope_scaling=d.get("rope_scaling"),
            max_position_embeddings=d.get("max_position_embeddings", 4096),
            attention_bias=d.get("attention_bias", False),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            bos_token_id=d.get("bos_token_id", -1),
            eos_token_ids=list(d.get("eos_token_ids", [])),
            weight_dtype=q.get("weight_dtype", "fp16"),
            lm_head_dtype=q.get("lm_head_dtype", "fp16"),
            group_size=q.get("group_size", 128),
            source=d.get("source", ""),
        )

    def validate(self) -> None:
        assert self.weight_dtype in QUANT_TYPES and self.lm_head_dtype in QUANT_TYPES
        assert self.num_heads % self.num_kv_heads == 0
        if "int4" in (self.weight_dtype, self.lm_head_dtype):
            assert self.group_size in (32, 64, 128), "group_size must be 32, 64 or 128"
            assert self.hidden_size % self.group_size == 0
            assert self.intermediate_size % self.group_size == 0
            assert self.q_size % self.group_size == 0


@dataclass
class TensorPlan:
    name: str
    dtype: str  # safetensors dtype string
    shape: tuple

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n * _ITEMSIZE[self.dtype]


def linear_plan(prefix: str, n: int, k: int, qtype: str, group_size: int, bias: bool):
    if qtype == "fp16":
        plan = [TensorPlan(f"{prefix}.weight", "F16", (n, k))]
    elif qtype == "int8":
        plan = [TensorPlan(f"{prefix}.weight", "I8", (n, k)), TensorPlan(f"{prefix}.scales", "F16", (n,))]
    elif qtype == "int4":
        g = k // group_size
        plan = [
            TensorPlan(f"{prefix}.weight", "U8", (n, k // 2)),
            TensorPlan(f"{prefix}.scales", "F16", (n, g)),
            TensorPlan(f"{prefix}.offsets", "F16", (n, g)),
        ]
    else:
        raise ValueError(qtype)
    if bias:
        plan.append(TensorPlan(f"{prefix}.bias", "F16", (n,)))
    return plan


def quantize_linear(w: torch.Tensor, qtype: str, group_size: int, clip_search: bool) -> list:
    """Returns tensors in the order of ``linear_plan`` (without bias)."""
    if qtype == "fp16":
        return [w.to(torch.float16)]
    if qtype == "int8":
        q, s = Q.quantize_int8(w, clip_search=clip_search)
        return [q, s]
    if qtype == "int4":
        p, s, o = Q.quantize_int4(w, group_size=group_size, clip_search=clip_search)
        return [p, s, o]
    raise ValueError(qtype)


class StreamingWriter:
    """Writes a safetensors file whose tensor list is known up front."""

    def __init__(self, path: str, plan: list[TensorPlan], metadata: dict[str, str]):
        self.path = path
        self.plan = plan
        header: dict = {"__metadata__": metadata}
        offset = 0
        for t in plan:
            header[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [offset, offset + t.nbytes]}
            offset += t.nbytes
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob += b" " * ((8 - len(blob) % 8) % 8)
        self._f = open(path, "wb")
        self._f.write(struct.pack("<Q", len(blob)))
        self._f.write(blob)
        self._index = 0

    def write(self, tensor: torch.Tensor) -> None:
        expected = self.plan[self._index]
        t = tensor.detach().contiguous().cpu()
        dtype = _ST_DTYPES[t.dtype]
        if dtype != expected.dtype or tuple(t.shape) != tuple(expected.shape):
            raise ValueError(
                f"tensor {expected.name}: got {dtype}{tuple(t.shape)}, planned {expected.dtype}{expected.shape}"
            )
        if t.dtype == torch.bfloat16:
            data = t.view(torch.int16).numpy().tobytes()
        else:
            data = t.numpy().tobytes()
        self._f.write(data)
        self._index += 1

    def close(self) -> None:
        if self._index != len(self.plan):
            raise RuntimeError(f"wrote {self._index} of {len(self.plan)} planned tensors")
        self._f.close()


def write_streaming(path: str, plan: list[TensorPlan], metadata: dict, producer: Iterable[torch.Tensor]):
    w = StreamingWriter(path, plan, metadata)
    for t in producer:
        w.write(t)
    w.close()


class RedlineFile:
    """Read access to a .rlm file (tensors returned as torch tensors)."""

    def __init__(self, path: str):
        from safetensors import safe_open

        self.path = path
        self._f = safe_open(path, framework="pt")
        meta = self._f.metadata() or {}
        if meta.get("format") != "redline" or "redline_config" not in meta:
            raise ValueError(f"{path} is not a Redline model file")
        self.spec = ModelSpec.from_json(json.loads(meta["redline_config"]))

    def keys(self) -> list[str]:
        return list(self._f.keys())

    def has(self, name: str) -> bool:
        return name in self._f.keys()

    def tensor(self, name: str) -> torch.Tensor:
        return self._f.get_tensor(name)

    def linear(self, prefix: str, qtype: str) -> torch.Tensor:
        """Dequantized fp32 weight [N, K]."""
        if qtype == "fp16":
            return self.tensor(f"{prefix}.weight").to(torch.float32)
        if qtype == "int8":
            return Q.dequantize_int8(self.tensor(f"{prefix}.weight"), self.tensor(f"{prefix}.scales"))
        return Q.dequantize_int4(
            self.tensor(f"{prefix}.weight"),
            self.tensor(f"{prefix}.scales"),
            self.tensor(f"{prefix}.offsets"),
            self.spec.group_size,
        )

    def to_hf_state_dict(self) -> dict[str, torch.Tensor]:
        """Dequantized weights in Hugging Face Llama naming (fp32)."""
        s = self.spec
        sd: dict[str, torch.Tensor] = {}
        f32 = lambda name: self.tensor(name).to(torch.float32)  # noqa: E731
        sd["model.embed_tokens.weight"] = f32("tok_embeddings.weight")
        for i in range(s.num_layers):
            p, hp = f"layers.{i}.", f"model.layers.{i}."
            sd[hp + "input_layernorm.weight"] = f32(p + "attn_norm.weight")
            sd[hp + "post_attention_layernorm.weight"] = f32(p + "ffn_norm.weight")
            qkv = self.linear(p + "wqkv", s.weight_dtype)
            q, k, v = torch.split(qkv, [s.q_size, s.kv_size, s.kv_size], dim=0)
            sd[hp + "self_attn.q_proj.weight"] = q
            sd[hp + "self_attn.k_proj.weight"] = k
            sd[hp + "self_attn.v_proj.weight"] = v
            if self.has(p + "wqkv.bias"):
                qb, kb, vb = torch.split(f32(p + "wqkv.bias"), [s.q_size, s.kv_size, s.kv_size])
                sd[hp + "self_attn.q_proj.bias"] = qb
                sd[hp + "self_attn.k_proj.bias"] = kb
                sd[hp + "self_attn.v_proj.bias"] = vb
            sd[hp + "self_attn.o_proj.weight"] = self.linear(p + "wo", s.weight_dtype)
            if self.has(p + "wo.bias"):
                sd[hp + "self_attn.o_proj.bias"] = f32(p + "wo.bias")
            gu = self.linear(p + "w_gate_up", s.weight_dtype)
            gate, up = torch.split(gu, [s.intermediate_size, s.intermediate_size], dim=0)
            sd[hp + "mlp.gate_proj.weight"] = gate
            sd[hp + "mlp.up_proj.weight"] = up
            sd[hp + "mlp.down_proj.weight"] = self.linear(p + "w_down", s.weight_dtype)
        sd["model.norm.weight"] = f32("norm.weight")
        if self.has("lm_head.weight"):
            sd["lm_head.weight"] = self.linear("lm_head", s.lm_head_dtype)
        else:
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
        return sd


def weight_bytes(path: str) -> int:
    """Total tensor bytes of a .rlm file (what the engine uploads to the GPU)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items() if k != "__metadata__")


def as_numpy_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()

