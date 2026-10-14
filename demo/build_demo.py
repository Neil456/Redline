#!/usr/bin/env python3
"""Builds demo/index.html from demo/template.html.

Injects code excerpts taken from the repository (so file paths and line
numbers in the demo always match the source) and, optionally, a real trace
written by `redline_bench --trace-out`:

    python demo/build_demo.py                       # simulations + code map only
    python demo/build_demo.py --trace runs/x/trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# key: (file, anchor regex, number of lines, caption)
EXCERPTS = {
    "scheduler": ("src/core/scheduler.cpp", r"auto schedule_decode = \[&\]\(\) \{", 29,
                  "Decode first: every running sequence gets one token; when the KV pool is empty the newest "
                  "sequence is preempted and recomputed later."),
    "block_alloc": ("src/core/block_allocator.cpp", r"int BlockAllocator::allocate\(\) \{", 15,
                    "KV blocks are plain integers handed out from a LIFO free list (recently freed blocks are "
                    "still warm in L2)."),
    "batch_add": ("src/core/batch_builder.cpp", r"int BatchBuilder::add\(", 24,
                  "Each token gets a position and a physical KV slot = block_table[pos / 16] * 16 + pos % 16."),
    "upload": ("src/cuda/engine/engine.cpp", r"DeviceBatch Engine::upload_batch\(", 13,
               "All step metadata is serialized into pinned memory and uploaded with one async copy."),
    "step_async": ("src/cuda/engine/engine.cpp", r"const int64_t prev = inflight_\.empty\(\)", 20,
                   "Async scheduling: launch step t+1, then retire step t. The GPU never waits on the host."),
    "launch": ("src/cuda/engine/engine.cpp", r"const int real_seqs = builder_->num_seqs\(\);", 25,
               "Pure-decode steps are padded to a batch bucket and replayed as a CUDA graph; sampled ids "
               "come back through a small D2H copy into pinned memory."),
    "capture": ("src/cuda/engine/engine.cpp", r"cudaGraphExec_t Engine::capture\(", 15,
                "Graph capture: one eager warm-up (lazy loading, cuBLASLt plans) then capture the whole "
                "forward + sampling."),
    "forward": ("src/cuda/model/llama.cpp", r"for \(int l = 0; l < c\.num_layers; \+\+l\) \{", 21,
                "One decoder layer: fused-QKV GEMM, RoPE + paged KV write, attention, O projection, "
                "fused residual RMSNorm, SwiGLU MLP."),
    "rope_kv": ("src/cuda/kernels/rope_kv.cu", r"if \(slot < 0\) continue;", 26,
                "RoPE is fused with the KV-cache write; INT8 KV stores one absmax scale per (token, head)."),
    "attn_decode": ("src/cuda/kernels/attention.cu", r"// Phase 1: scores\. One thread per key", 23,
                    "Decode attention: each CTA scores keys for all G query heads of a KV head, so every K "
                    "vector is read from memory once per GQA group."),
    "gemv_w4": ("src/cuda/kernels/gemv_weight_only.cu", r"for \(int kk = lane \* 32; kk < k; kk \+= 32 \* 32\) \{", 38,
                "W4A16 GEMV: one 16-byte load brings 32 weights; y = s * sum(x*q) + o * sum(x) applies the "
                "group scale and offset once per 32 values."),
    "linear_dispatch": ("src/cuda/gemm/linear.cpp", r"void LinearExecutor::forward\(", 45,
                        "The GEMM dispatcher: GEMV for decode, CUTLASS / cuBLASLt W8A8 / TensorRT-LLM / "
                        "dequant+cuBLASLt for larger batches."),
    "sampling": ("src/cuda/kernels/sampling.cu", r"__device__ uint32_t radix_select_topk\(", 30,
                 "Top-k without sorting the vocabulary: 4 radix passes over order-preserving float keys find "
                 "the exact k-th largest logit."),
    "gumbel": ("src/cuda/kernels/sampling.cu", r"__device__ int gumbel_sample\(", 11,
               "Sampling = argmax(logit/T + Gumbel noise) over the kept set; Philox keyed by (seed, position) "
               "makes it independent of batch composition."),
    "spec_step": ("src/cuda/engine/engine.cpp", r"// Draft round 0 catches the draft KV cache up", 33,
                  "Speculative step: the draft proposes k tokens (graph-captured rounds); the target verifies "
                  "all k+1 positions in one pass."),
    "spec_verify": ("src/cuda/kernels/sampling.cu", r"for \(int i = 0; i < num_draft; \+\+i\) \{", 25,
                    "Verification: accept with probability min(1, p_target / p_draft), resample the first "
                    "rejection from max(0, p_target - p_draft). Exact for greedy, lossless for sampling."),
    "harness": ("src/cuda/bench/harness.cpp", r"// ---- per-step metrics", 26,
                "Benchmark metrics: decode throughput from step wall time, GPU time from CUDA events, "
                "host overhead per step."),
    "quant_int4": ("python/redline/quantize.py", r"def quantize_int4\(", 41,
                   "INT4 export: asymmetric groups of 128 with a per-group clipping search that minimizes "
                   "reconstruction error."),
    "pinned_upload": ("src/cuda/model/weights.cpp", r"void upload\(void\* dst, const void\* src, size_t bytes\) \{", 15,
                      "Weight loading streams through two pinned staging buffers, so the host memcpy overlaps "
                      "the previous chunk's DMA."),
}


def extract(path: str, anchor: str, count: int) -> dict:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        if re.search(anchor, line):
            chunk = lines[i : i + count]
            indent = min((len(l) - len(l.lstrip()) for l in chunk if l.strip()), default=0)
            return {"file": path, "start": i + 1, "end": i + len(chunk),
                    "code": "\n".join(l[indent:] for l in chunk).rstrip()}
    raise SystemExit(f"anchor not found in {path}: {anchor}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="trace JSON from redline_bench --trace-out to embed")
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "index.html"))
    ap.add_argument("--fragment", action="store_true",
                    help="write the page body only (for publishing as a hosted artifact)")
    args = ap.parse_args(argv)

    excerpts = {}
    for key, (path, anchor, count, caption) in EXCERPTS.items():
        e = extract(path, anchor, count)
        e["caption"] = caption
        excerpts[key] = e
        print(f"  {key:16s} {path}:{e['start']}-{e['end']}", file=sys.stderr)
    trace = None
    if args.trace:
        with open(args.trace) as f:
            trace = json.load(f)
        if trace.get("format") != "redline-trace-v1":
            raise SystemExit(f"{args.trace} is not a redline trace (missing format: redline-trace-v1)")
    with open(os.path.join(ROOT, "demo", "template.html"), encoding="utf-8") as f:
        html = f.read()

    def js(obj) -> str:  # safe inside <script>
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    html = html.replace("/*__EXCERPTS__*/null", js(excerpts)).replace("/*__TRACE__*/null", js(trace))
    if not args.fragment:  # standalone document that opens straight from the repo
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KiB, trace={'yes' if trace else 'no'})", file=sys.stderr)


if __name__ == "__main__":
    main()
