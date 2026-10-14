"""Run a workload suite (see redline.workloads) across Redline configurations
and the HF / vLLM baselines, writing one JSON report per (system, workload).

    python -m redline.suite --workloads runs/x/workloads --out runs/x/results \\
        --redline fp16=runs/x/models/target-fp16.rlm \\
        --redline int8=runs/x/models/target-int8.rlm \\
        --redline int4=runs/x/models/target-int4.rlm \\
        --redline "int4_spec=runs/x/models/target-int4.rlm --draft-model runs/x/models/draft-int4.rlm --spec-tokens 4" \\
        --hf meta-llama/Llama-3.1-8B-Instruct --vllm meta-llama/Llama-3.1-8B-Instruct

Workloads are grouped by batch size so each engine instance is configured
with max_seqs = batch size and a KV cache sized for exactly that batch, so
peak VRAM reflects what the workload needs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict

from .runtime import find_binary


def group_workloads(wdir: str) -> dict:
    with open(os.path.join(wdir, "manifest.json")) as f:
        manifest = json.load(f)
    groups: dict = defaultdict(list)
    for e in manifest["workloads"]:
        s = e["settings"]
        groups[(s["kind"], int(s.get("batch_size", 32)))].append(os.path.join(wdir, e["path"]))
    return groups


def sizes_for(paths: list[str], batch: int, spec_tokens: int) -> tuple[int, int]:
    """(max_seq_len, kv_cache_tokens) just large enough for these workloads."""
    longest = 0
    for p in paths:
        with open(p) as f:
            w = json.load(f)
        for r in w["requests"]:
            prompt = len(r.get("prompt_ids", [])) or r.get("prompt_len", 0)
            longest = max(longest, prompt + r["max_new_tokens"])
    max_seq_len = ((longest + spec_tokens + 2 + 255) // 256) * 256
    return max_seq_len, batch * max_seq_len + 256


def run(cmd: list[str], dry: bool) -> None:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), file=sys.stderr, flush=True)
    if not dry:
        rc = subprocess.run(cmd).returncode
        if rc != 0:  # keep going: one OOM configuration should not abort the suite
            print(f"! command failed with exit code {rc}", file=sys.stderr, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--redline", action="append", default=[], help="TAG=MODEL.rlm [extra redline_bench flags]")
    ap.add_argument("--hf", help="HF model id for the Transformers baseline")
    ap.add_argument("--hf-quant", action="append", default=[], help="extra HF baselines: bnb8, bnb4")
    ap.add_argument("--vllm", help="HF model id for the vLLM baseline")
    ap.add_argument("--vllm-python", default=sys.executable,
                    help="interpreter for the vLLM baseline (vLLM pins its own torch, so it often lives in a venv)")
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-seq-len", type=int, default=4096, help="vLLM max_model_len")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    groups = group_workloads(args.workloads)
    bench = find_binary("redline_bench") if not args.dry_run else "redline_bench"
    for spec in args.redline:
        tag, rest = spec.split("=", 1)
        parts = shlex.split(rest)
        model, extra = parts[0], parts[1:]
        spec_tokens = 0
        if "--spec-tokens" in extra:
            spec_tokens = int(extra[extra.index("--spec-tokens") + 1])
        elif "--draft-model" in extra:
            spec_tokens = 4  # engine default
        for (kind, batch), paths in sorted(groups.items()):
            max_seq_len, kv_tokens = sizes_for(paths, batch, spec_tokens)
            cmd = [bench, "--model", model, *extra, "--tag", f"redline_{tag}", "--out-dir", args.out,
                   "--max-seqs", str(batch), "--max-seq-len", str(max_seq_len),
                   "--kv-cache-tokens", str(kv_tokens), "--track-memory"]
            if kind == "online":
                cmd.append("--online")
            for p in paths:
                cmd += ["--workload", p]
            run(cmd, args.dry_run)

    py = sys.executable
    if args.hf:
        all_paths = [p for ps in groups.values() for p in ps]
        for quant in ["fp16", *args.hf_quant]:
            cmd = [py, "-m", "redline.baselines.hf_bench", "--model", args.hf, "--quant", quant, "--out", args.out]
            for p in all_paths:
                cmd += ["--workload", p]
            run(cmd, args.dry_run)
    if args.vllm:
        for (kind, batch), paths in sorted(groups.items()):
            cmd = [args.vllm_python, "-m", "redline.baselines.vllm_bench", "--model", args.vllm, "--out", args.out,
                   "--max-num-seqs", str(batch), "--max-model-len", str(args.max_seq_len),
                   "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization)]
            for p in paths:
                cmd += ["--workload", p]
            run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
