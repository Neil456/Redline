#!/usr/bin/env python3
"""Builds demo/index.html from demo/template.html.

Injects code excerpts taken from the repository (so file paths and line
numbers in the demo always match the source), the measured RTX 4090 results
in results/rtx4090 and the traces in demo/traces (written by
`redline_bench --trace-out`):

    python demo/build_demo.py                                   # everything that exists in the repo
    python demo/build_demo.py --trace "My run=runs/x/trace.json"  # replace the traces
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
    "forward": ("src/cuda/model/llama.cpp", r"for \(int l = 0; l < c\.num_layers; \+\+l\) \{", 28,
                "One decoder layer: fused-QKV GEMM, RoPE + paged KV write, attention, O projection, "
                "fused residual RMSNorm, gate/up GEMM with SwiGLU in its epilogue, down projection. mark() feeds the "
                "per-op profiler."),
    "rope_kv": ("src/cuda/kernels/rope_kv.cu", r"if \(slot < 0\) continue;", 26,
                "RoPE is fused with the KV-cache write; INT8 KV stores one absmax scale per (token, head)."),
    "attn_decode": ("src/cuda/kernels/attention.cu", r"// Phase 1: scores\. One thread per key", 23,
                    "Decode attention: each CTA scores keys for all G query heads of a KV head, so every K "
                    "vector is read from memory once per GQA group."),
    "gemv_w4": ("src/cuda/kernels/gemv_weight_only.cu", r"for \(int kk = lane \* 32; kk < k; kk \+= 32 \* 32\) \{", 46,
                "W4A16 GEMV: one 16-byte load brings 32 weights, each unpacked once and reused for every "
                "activation row; y = s * sum(x*q) + o * sum(x) applies the group scale and offset once per 32 values."),
    "gemm_tc": ("src/cuda/kernels/gemm_w4a16_tc.cu", r"// Software pipeline: the next chunk's weights", 59,
                "Tensor-core W4A16 main loop: the next chunk's INT4 weights load while this chunk computes; each "
                "16-byte load is a lane's B fragments for eight n8 tiles, dequantized with one hfma2 and fed "
                "to mma.sync.m16n8k16 for up to 64 rows."),
    "dequant_tc": ("src/cuda/kernels/gemm_w4a16_tc.cu", r"// 8 nibbles -> 4 half2", 12,
                   "INT4 -> fp16 without conversions: lop3 drops each nibble into the mantissa of 1024.0 "
                   "(0x6400), then one subtract (or a multiply-add for the high nibble) leaves the exact value."),
    "linear_dispatch": ("src/cuda/gemm/linear.cpp", r"void LinearExecutor::forward\(", 45,
                        "The GEMM dispatcher: GEMV for decode, tensor-core INT4 up to 64 rows, CUTLASS / "
                        "cuBLASLt W8A8 / TensorRT-LLM / dequant+cuBLASLt for larger batches."),
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
    "gptq": ("python/redline/calibrate.py", r"    for i1 in range\(0, k, blocksize\):", 23,
             "GPTQ: quantize one column at a time and push its rounding error onto the columns not yet "
             "quantized, weighted by the inverse Hessian of the layer's calibration inputs."),
    "awq": ("python/redline/calibrate.py", r"def awq_search\(", 16,
            "AWQ: scale input channels by mean|x|^alpha before quantizing (and fold 1/s into the previous op), "
            "choosing alpha by the layer's output error on calibration activations."),
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


# Traces recorded on the RTX 4090 run (Llama 3.1 8B, online workload: Poisson arrivals at 4 req/s).
DEFAULT_TRACES = [
    ("INT4 weights", "demo/traces/int4_online.json",
     "W4A16 GEMV for decode, default KV pool. The batch fills to ~29 sequences and the queue never builds."),
    ("INT8, small KV pool", "demo/traces/int8_online_kvtight.json",
     "KV pool capped at 12,288 tokens (768 blocks) to show admission control: the pool hits 100%, requests "
     "queue, and the scheduler preempts and recomputes the newest sequences."),
    ("INT4 + speculative", "demo/traces/int4_spec_online.json",
     "Llama 3.2 1B INT4 draft proposes 4 tokens per step; the target verifies all 5 positions in one pass."),
]

# Bytes streamed per decoded token at batch 1 (weight files minus the embedding table, which is
# gathered, not streamed) and the RTX 4090's DRAM bandwidth.
WEIGHT_BYTES = {"redline_fp16": 15.01e9, "redline_int8": 7.51e9, "redline_int4": 4.23e9}
DRAM_BW = 1008e9

SYSTEMS = [  # tag, label, is_redline
    ("redline_fp16", "Redline FP16", True), ("redline_int8", "Redline INT8", True),
    ("redline_int4", "Redline INT4", True), ("hf_fp16", "HF Transformers FP16", False),
    ("vllm", "vLLM FP16", False), ("vllm_awq_int4", "vLLM AWQ INT4", False),
]


def load_results(root: str) -> dict | None:
    """Summary of results/rtx4090 for the demo, computed with the same code as redline.report."""
    if not os.path.isdir(os.path.join(root, "sweep")):
        return None
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from redline import report  # noqa: E402

    sweep, quick = report.load_reports(os.path.join(root, "sweep")), report.load_reports(os.path.join(root, "quick"))
    batches = [1, 2, 4, 8, 16, 32]
    series = []
    for tag, label, ours in SYSTEMS:
        reps = [sweep.get(tag, {}).get(f"sweep_b{b}_in512_out256") for b in batches]
        if any(r is None for r in reps):
            continue
        series.append({"tag": tag, "label": label, "redline": ours,
                       "tps": [round(report.decode_tps(r), 1) for r in reps],
                       "tpot": [round(r["tpot_ms"]["p50"], 2) for r in reps],
                       "ttft": [round(r["ttft_ms"]["p50"], 1) for r in reps]})
    vs_hf = {}
    for tag, label, ours in SYSTEMS:
        c = report.compare(sweep, tag, "hf_fp16") if tag != "hf_fp16" else None
        q = report.compare(quick, tag, "hf_fp16") if tag != "hf_fp16" else None
        if c:
            vs_hf[tag] = {"decode": round(c["decode_speedup_geomean"], 3),
                          "vram": None if not q or q["peak_vram_reduction_mean"] is None
                          else round(q["peak_vram_reduction_mean"], 3)}
    bandwidth = []
    for tag, nbytes in WEIGHT_BYTES.items():
        r = quick.get(tag, {}).get("offline_b1_in128_out128_greedy")
        if r:
            floor = nbytes / DRAM_BW * 1e3
            bandwidth.append({"tag": tag, "tpot": round(r["tpot_ms"]["p50"], 2), "floor": round(floor, 2),
                              "gb": round(nbytes / 1e9, 2)})
    quality = []
    qpath = os.path.join(root, "quality.json")
    if os.path.exists(qpath):
        with open(qpath) as f:
            q = json.load(f)
        ref = q.get("reference", {}).get("results", {})
        for c in q["candidates"]:
            quality.append({"model": os.path.basename(c["model"]), "retention": round(c["retention"]["mean"], 4),
                            "per_task": {k: round(v, 4) for k, v in c["retention"]["per_task"].items()},
                            "ppl": round(c["results"]["wikitext"]["perplexity"], 3) if "wikitext" in c["results"] else None})
        if ref.get("wikitext"):
            quality.insert(0, {"model": "HF FP16 (reference)", "retention": 1.0, "per_task": {},
                               "ppl": round(ref["wikitext"]["perplexity"], 3)})
    env = ""
    if os.path.exists(os.path.join(root, "environment.txt")):
        with open(os.path.join(root, "environment.txt")) as f:
            env = f.readline().strip()
    return {"batches": batches, "series": series, "vs_hf": vs_hf, "bandwidth": bandwidth,
            "quality": quality, "env": env, "source": os.path.relpath(root, ROOT)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", action="append", default=[],
                    help="LABEL=PATH (or PATH) of a redline_bench --trace-out file; repeatable. "
                         "Default: the traces in demo/traces")
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "rtx4090"),
                    help="measured results directory (written by redline_bench / the baselines)")
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
    specs = []
    for t in args.trace:
        label, _, path = t.rpartition("=") if "=" in t else ("", "", t)
        specs.append((label or os.path.basename(path), path, ""))
    if not specs:
        specs = [(lab, os.path.join(ROOT, p), note) for lab, p, note in DEFAULT_TRACES
                 if os.path.exists(os.path.join(ROOT, p))]
    traces = []
    for label, path, note in specs:
        with open(path) as f:
            tr = json.load(f)
        if tr.get("format") != "redline-trace-v1":
            raise SystemExit(f"{path} is not a redline trace (missing format: redline-trace-v1)")
        traces.append({"label": label, "note": note, "trace": tr})
    results = load_results(args.results)
    with open(os.path.join(ROOT, "demo", "template.html"), encoding="utf-8") as f:
        html = f.read()

    def js(obj) -> str:  # safe inside <script>
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    html = (html.replace("/*__EXCERPTS__*/null", js(excerpts)).replace("/*__TRACE__*/null", js(traces or None))
            .replace("/*__RESULTS__*/null", js(results)))
    if not args.fragment:  # standalone document that opens straight from the repo
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KiB, traces={len(traces)}, "
          f"results={'yes' if results else 'no'})", file=sys.stderr)


if __name__ == "__main__":
    main()
