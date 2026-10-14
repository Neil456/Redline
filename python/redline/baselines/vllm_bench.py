"""vLLM baseline on the same workloads as redline_bench.

Requests are submitted to vLLM's async engine (continuous batching) with the
workload's arrival times; every streamed output is timestamped, giving
TTFT / inter-token latency. Device memory is sampled with NVML. vLLM
pre-allocates its KV cache from --gpu-memory-utilization, so peak VRAM
reflects that setting; use the smallest value that serves the workload for a
like-for-like memory comparison.

    python -m redline.baselines.vllm_bench --model meta-llama/Llama-3.1-8B-Instruct \\
        --workload workloads/offline_b8_in512_out256_greedy.json --out results/vllm/ --max-num-seqs 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid

from ..metrics import DeviceMemorySampler, RequestTrace, build_report, decode_window


async def _run_request(engine, req: dict, t0: float, online: bool, sampling_cls) -> RequestTrace:
    if online:
        await asyncio.sleep(max(0.0, t0 + req["arrival_time"] - time.perf_counter()))
    sp = sampling_cls(
        max_tokens=req["max_new_tokens"],
        min_tokens=req["max_new_tokens"],
        ignore_eos=True,
        temperature=req["temperature"],
        top_p=req["top_p"],
        top_k=req["top_k"] if req["top_k"] > 0 else -1,
        seed=req["seed"] if req["temperature"] > 0 else None,
    )
    trace = RequestTrace(req["id"], len(req["prompt_ids"]), time.perf_counter())
    seen = 0
    async for out in engine.generate({"prompt_token_ids": req["prompt_ids"]}, sp, request_id=str(uuid.uuid4())):
        n = len(out.outputs[0].token_ids)
        now = time.perf_counter()
        trace.token_times.extend([now] * (n - seen))
        seen = n
    return trace


async def run_workload(engine, w: dict, sampling_cls, warmup: bool = True) -> dict:
    online = w.get("settings", {}).get("kind") == "online"
    if warmup:
        await _run_request(engine, dict(w["requests"][0], max_new_tokens=8), time.perf_counter(), False,
                           sampling_cls)
    with DeviceMemorySampler() as mem:
        t0 = time.perf_counter()
        traces = await asyncio.gather(*[_run_request(engine, r, t0, online, sampling_cls) for r in w["requests"]])
        duration = time.perf_counter() - t0
    n, dt = decode_window(list(traces))
    return build_report("vllm", w, list(traces), duration, n, dt, mem.peak)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", default=None, help="e.g. awq, gptq, fp8 (needs a matching checkpoint)")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model, dtype="float16", quantization=args.quantization, max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager))
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or ("vllm" + (f"_{args.quantization}" if args.quantization else ""))

    async def run_all():
        for path in args.workload:
            with open(path) as f:
                w = json.load(f)
            rep = await run_workload(engine, w, SamplingParams)
            rep["tag"] = tag
            rep["vllm_settings"] = vars(args)
            with open(os.path.join(args.out, f"{tag}__{w['name']}.json"), "w") as f:
                json.dump(rep, f, indent=2)
            print(f"{tag} {w['name']}: decode {rep['decode_throughput_tok_s']:.1f} tok/s, "
                  f"TTFT p50 {rep['ttft_ms']['p50']:.1f} ms, ITL p50 {rep['itl_ms']['p50']:.2f} ms, "
                  f"peak {rep['memory']['peak_device_used_bytes'] / 2**30:.2f} GiB", file=sys.stderr)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
