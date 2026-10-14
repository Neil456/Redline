"""Hugging Face Transformers baseline on the same workloads as redline_bench.

Requests are served in static batches of the workload's batch_size with
left padding (the standard `model.generate` path). A streamer timestamps every
generated token after a CUDA sync, giving TTFT / inter-token latency; device
memory is sampled with NVML.

    python -m redline.baselines.hf_bench --model meta-llama/Llama-3.1-8B-Instruct \\
        --workload workloads/offline_b8_in512_out256_greedy.json --out results/hf/

Use --quant bnb8 / bnb4 for bitsandbytes INT8/INT4 baselines and
--static-cache to enable the static KV cache + torch.compile path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from ..metrics import DeviceMemorySampler, RequestTrace, build_report, decode_window


class TimingStreamer:
    """transformers streamer that records a timestamp per generation step."""

    def __init__(self):
        self.times: list[float] = []
        self._first = True

    def put(self, value):
        if self._first:  # generate() first passes the prompt
            self._first = False
            return
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())

    def end(self):
        pass


def load_model(model_id: str, quant: str, attn: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"dtype": torch.float16, "device_map": "cuda", "attn_implementation": attn}
    if quant in ("bnb8", "bnb4"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if quant == "bnb8" else
                                         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.inference_mode()
def run_batch(model, tok, reqs: list[dict], static_cache: bool):
    max_len = max(len(r["prompt_ids"]) for r in reqs)
    pad = tok.pad_token_id
    ids = torch.full((len(reqs), max_len), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(reqs):  # left padding
        p = torch.tensor(r["prompt_ids"])
        ids[i, max_len - len(p):] = p
        mask[i, max_len - len(p):] = 1
    max_new = max(r["max_new_tokens"] for r in reqs)
    r0 = reqs[0]
    gen_kwargs = dict(
        max_new_tokens=max_new,
        min_new_tokens=max_new,  # ignore EOS, like the other systems
        do_sample=r0["temperature"] > 0,
        pad_token_id=pad,
    )
    if r0["temperature"] > 0:
        gen_kwargs.update(temperature=r0["temperature"], top_p=r0["top_p"], top_k=r0["top_k"] or None)
    if static_cache:
        gen_kwargs["cache_implementation"] = "static"
    streamer = TimingStreamer()
    torch.cuda.synchronize()
    arrival = time.perf_counter()
    model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), streamer=streamer, **gen_kwargs)
    torch.cuda.synchronize()
    traces = []
    for r in reqs:
        t = RequestTrace(r["id"], len(r["prompt_ids"]), arrival)
        t.token_times = streamer.times[: r["max_new_tokens"]]
        traces.append(t)
    return traces


def run_workload(model, tok, w: dict, static_cache: bool, warmup: bool = True) -> dict:
    reqs = w["requests"]
    bs = int(w.get("settings", {}).get("batch_size", len(reqs)))
    if warmup:
        run_batch(model, tok, [dict(reqs[0], max_new_tokens=8)], static_cache)
    torch.cuda.reset_peak_memory_stats()
    traces = []
    decode_tokens, decode_time = 0, 0.0
    oom_splits = 0

    def run_chunk(chunk: list[dict]) -> list[list]:
        """Runs one static batch; on OOM, halves it (padded prefill of long prompts is the usual cause)."""
        nonlocal oom_splits
        try:
            return [run_batch(model, tok, chunk, static_cache)]
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            torch.cuda.empty_cache()
            oom_splits += 1
            print(f"note: OOM at batch {len(chunk)}, splitting in half", file=sys.stderr)
            h = len(chunk) // 2
            return run_chunk(chunk[:h]) + run_chunk(chunk[h:])

    with DeviceMemorySampler() as mem:
        t0 = time.perf_counter()
        for i in range(0, len(reqs), bs):
            for batch_traces in run_chunk(reqs[i : i + bs]):
                n, dt = decode_window(batch_traces)
                decode_tokens += n
                decode_time += dt
                traces += batch_traces
        duration = time.perf_counter() - t0
    return build_report("hf", w, traces, duration, decode_tokens, decode_time, mem.peak,
                        {"torch_max_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "oom_batch_splits": oom_splits})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", action="append", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--quant", default="fp16", choices=["fp16", "bnb8", "bnb4"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--static-cache", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    if any(json.load(open(p)).get("settings", {}).get("kind") == "online" for p in args.workload):
        print("note: HF baseline runs online workloads as static batches (no continuous batching)", file=sys.stderr)
    model, tok = load_model(args.model, args.quant, args.attn)
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"hf_{args.quant}" + ("_static" if args.static_cache else "")
    for path in args.workload:
        with open(path) as f:
            w = json.load(f)
        rep = run_workload(model, tok, w, args.static_cache)
        rep["tag"] = tag
        with open(os.path.join(args.out, f"{tag}__{w['name']}.json"), "w") as f:
            json.dump(rep, f, indent=2)
        print(f"{tag} {w['name']}: decode {rep['decode_throughput_tok_s']:.1f} tok/s, "
              f"TTFT p50 {rep['ttft_ms']['p50']:.1f} ms, ITL p50 {rep['itl_ms']['p50']:.2f} ms, "
              f"peak {rep['memory']['peak_device_used_bytes'] / 2**30:.2f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
