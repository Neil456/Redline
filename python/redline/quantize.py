"""Weight-only quantization used by the exporter.

INT8: symmetric, per output channel.      w ~= q * scale[n],          q in [-127, 127]
INT4: asymmetric, groups along K.         w ~= q * scale[n, g] + offset[n, g], q in [0, 15]
      Packed two per byte, even column in the low nibble.

Both optionally search a clipping ratio per channel/group that minimizes the
reconstruction MSE (cheap and noticeably better than plain round-to-nearest
for INT4). Scales are rounded to fp16 *before* computing q so the stored
values reproduce the dequantized weights exactly.
"""

from __future__ import annotations

import torch

INT8_CLIP_RATIOS = (1.0, 0.98, 0.96, 0.94, 0.92, 0.9, 0.88, 0.85)
INT4_CLIP_RATIOS = (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.7)
_MIN_SCALE = 1e-7


def _fp16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def quantize_int8(w: torch.Tensor, clip_search: bool = True, chunk_rows: int = 4096):
    """Returns (q int8 [N, K], scales fp16 [N])."""
    assert w.dim() == 2
    n, _ = w.shape
    q_out = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    s_out = torch.empty(n, dtype=torch.float16, device=w.device)
    ratios = INT8_CLIP_RATIOS if clip_search else (1.0,)
    for r0 in range(0, n, chunk_rows):
        x = w[r0 : r0 + chunk_rows].to(torch.float32)
        amax = x.abs().amax(dim=1)
        best_err = torch.full_like(amax, float("inf"))
        best_scale = torch.ones_like(amax)
        for ratio in ratios:
            scale = _fp16_round(torch.clamp(amax * ratio / 127.0, min=_MIN_SCALE))
            q = torch.clamp(torch.round(x / scale[:, None]), -127, 127)
            err = ((q * scale[:, None] - x) ** 2).sum(dim=1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, scale, best_scale)
        best_scale = torch.where(amax > 0, best_scale, torch.ones_like(best_scale))
        q = torch.clamp(torch.round(x / best_scale[:, None]), -127, 127)
        q_out[r0 : r0 + chunk_rows] = q.to(torch.int8)
        s_out[r0 : r0 + chunk_rows] = best_scale.to(torch.float16)
    return q_out, s_out


def quantize_int4(
    w: torch.Tensor, group_size: int = 128, clip_search: bool = True, chunk_rows: int = 1024
):
    """Returns (packed uint8 [N, K/2], scales fp16 [N, K/g], offsets fp16 [N, K/g])."""
    assert w.dim() == 2
    n, k = w.shape
    assert k % group_size == 0, f"K={k} not divisible by group_size={group_size}"
    groups = k // group_size
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
    scales = torch.empty((n, groups), dtype=torch.float16, device=w.device)
    offsets = torch.empty((n, groups), dtype=torch.float16, device=w.device)
    ratios = INT4_CLIP_RATIOS if clip_search else (1.0,)
    for r0 in range(0, n, chunk_rows):
        x = w[r0 : r0 + chunk_rows].to(torch.float32).reshape(-1, groups, group_size)
        lo = x.amin(dim=2)
        hi = x.amax(dim=2)
        best_err = torch.full_like(lo, float("inf"))
        best_s = torch.ones_like(lo)
        best_o = torch.zeros_like(lo)
        for ratio in ratios:
            # Shrink both range endpoints toward zero (AWQ-style clipping).
            lo_r, hi_r = lo * ratio, hi * ratio
            s = _fp16_round(torch.clamp((hi_r - lo_r) / 15.0, min=_MIN_SCALE))
            o = _fp16_round(lo_r)
            q = torch.clamp(torch.round((x - o[..., None]) / s[..., None]), 0, 15)
            err = ((q * s[..., None] + o[..., None] - x) ** 2).sum(dim=2)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s = torch.where(better, s, best_s)
            best_o = torch.where(better, o, best_o)
        flat = hi <= lo  # constant group: represent exactly with q = 0
        best_s = torch.where(flat, torch.ones_like(best_s), best_s)
        best_o = torch.where(flat, _fp16_round(lo), best_o)
        q = torch.clamp(torch.round((x - best_o[..., None]) / best_s[..., None]), 0, 15)
        q = q.reshape(x.shape[0], k).to(torch.uint8)
        packed[r0 : r0 + chunk_rows] = q[:, 0::2] | (q[:, 1::2] << 4)
        scales[r0 : r0 + chunk_rows] = best_s.to(torch.float16)
        offsets[r0 : r0 + chunk_rows] = best_o.to(torch.float16)
    return packed, scales, offsets


def dequantize_int8(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scales.to(torch.float32)[:, None]


def dequantize_int4(
    packed: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor, group_size: int
) -> torch.Tensor:
    n = packed.shape[0]
    k = packed.shape[1] * 2
    q = torch.empty((n, k), dtype=torch.float32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.float32)
    q[:, 1::2] = (packed >> 4).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(group_size, dim=1)
    o = offsets.to(torch.float32).repeat_interleave(group_size, dim=1)
    return q * s + o
