"""Aggregate benchmark JSON reports (Redline, HF, vLLM) into comparison tables.

    python -m redline.report --results results/ --baseline hf_fp16 --quality quality.json --out REPORT.md

Every report file carries a `tag` (system/config) and a `workload` name. For
each Redline tag the report computes, over the workloads it shares with the
baseline tag, the geometric-mean decode-throughput ratio and the mean peak
VRAM reduction. Decode throughput is compared on an ITL basis
(batch_size x 1000 / ITL p50) so all systems use the same definition.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def load_reports(root: str) -> dict:
    """{tag: {workload: report}}"""
    out: dict = defaultdict(dict)
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        try:
            with open(path) as f:
                r = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        items = r if isinstance(r, list) else [r]
        for rep in items:
            if isinstance(rep, dict) and "workload" in rep and "ttft_ms" in rep:
                out[rep.get("tag", rep.get("system", "?"))][rep["workload"]] = rep
    return out


def decode_tps(rep: dict) -> float:
    s = rep.get("workload_settings") or {}
    itl = rep["itl_ms"]["p50"]
    if s.get("kind", "offline") == "offline" and itl > 0 and "batch_size" in s:
        return s["batch_size"] * 1000.0 / itl
    return rep.get("decode_throughput_tok_s", 0.0)


def peak_gib(rep: dict) -> float:
    return rep["memory"]["peak_device_used_bytes"] / 2**30


def compare(reports: dict, tag: str, baseline: str) -> dict | None:
    shared = sorted(set(reports[tag]) & set(reports.get(baseline, {})))
    if not shared:
        return None
    ratios, mem = [], []
    for w in shared:
        a, b = reports[tag][w], reports[baseline][w]
        if decode_tps(b) > 0 and decode_tps(a) > 0:
            ratios.append(decode_tps(a) / decode_tps(b))
        if peak_gib(b) > 0:
            mem.append(1.0 - peak_gib(a) / peak_gib(b))
    if not ratios:
        return None
    return {
        "workloads": len(shared),
        "decode_speedup_geomean": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
        "decode_speedup_min": min(ratios),
        "decode_speedup_max": max(ratios),
        "peak_vram_reduction_mean": sum(mem) / len(mem) if mem else 0.0,
    }


def fmt_table(reports: dict, workload: str, tags: list[str]) -> str:
    lines = [f"#### {workload}", "",
             "| system | decode tok/s | TTFT p50 / p99 (ms) | ITL p50 / p99 (ms) | peak VRAM (GiB) | accept |",
             "|---|---:|---:|---:|---:|---:|"]
    for t in tags:
        r = reports[t].get(workload)
        if r is None:
            continue
        acc = r.get("speculative", {})
        acc_s = f"{acc.get('acceptance_rate', 0):.2f}" if acc.get("enabled") else "-"
        lines.append(f"| {t} | {decode_tps(r):.1f} | {r['ttft_ms']['p50']:.1f} / {r['ttft_ms']['p99']:.1f} | "
                     f"{r['itl_ms']['p50']:.2f} / {r['itl_ms']['p99']:.2f} | {peak_gib(r):.2f} | {acc_s} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--baseline", action="append", default=None, help="baseline tag(s), e.g. hf_fp16, vllm")
    ap.add_argument("--quality", help="output of redline.eval_quality")
    ap.add_argument("--out", default=None, help="markdown output (default: stdout)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    reports = load_reports(args.results)
    tags = sorted(reports)
    baselines = args.baseline or [t for t in tags if t.startswith(("hf", "vllm"))]
    redline_tags = [t for t in tags if t not in baselines]
    md = ["# Redline benchmark report", ""]
    summary = {"comparisons": {}}
    md += ["## Headline", "", "| config | vs | workloads | decode speedup (geomean) | range | peak VRAM reduction |",
           "|---|---|---:|---:|---:|---:|"]
    for t in redline_tags:
        for b in baselines:
            c = compare(reports, t, b)
            if c is None:
                continue
            summary["comparisons"][f"{t}__vs__{b}"] = c
            md.append(f"| {t} | {b} | {c['workloads']} | {(c['decode_speedup_geomean'] - 1) * 100:+.1f}% | "
                      f"{c['decode_speedup_min']:.2f}x-{c['decode_speedup_max']:.2f}x | "
                      f"{c['peak_vram_reduction_mean'] * 100:.1f}% |")
    if args.quality:
        with open(args.quality) as f:
            q = json.load(f)
        md += ["", "## Evaluation quality (vs FP16 Hugging Face)", "", "| model | retention | per task |", "|---|---:|---|"]
        for c in q["candidates"]:
            per = ", ".join(f"{k} {v * 100:.1f}%" for k, v in c["retention"]["per_task"].items())
            md.append(f"| {os.path.basename(c['model'])} | {c['retention']['mean'] * 100:.2f}% | {per} |")
        summary["quality"] = {c["model"]: c["retention"] for c in q["candidates"]}
    md += ["", "## Per-workload results", ""]
    workloads = sorted({w for t in tags for w in reports[t]})
    for w in workloads:
        md.append(fmt_table(reports, w, tags))
    text = "\n".join(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
