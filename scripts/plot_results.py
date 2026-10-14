#!/usr/bin/env python3
"""Renders the README figures (static SVG, light + dark) from results/rtx4090.

    python scripts/plot_results.py            # writes docs/img/*.svg

No plotting dependencies: the SVG is written directly so the figures match the
README's light and dark themes exactly. Every number comes from the benchmark
JSON through redline.report, the same code that writes REPORT.md.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from xml.sax.saxutils import escape

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from redline import report  # noqa: E402

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

# Text tokens tuned for GitHub's light (#ffffff) and dark (#0d1117) page surfaces; series hues are
# categorical slots 1-3 of the validated default palette (checked against both surfaces).
THEMES = {
    "light": {"surface": "#ffffff", "text": "#1f2328", "text2": "#59636e", "muted": "#818b98",
              "grid": "#e6e9ed", "axis": "#c8d0d8", "blue": "#2a78d6", "orange": "#eb6834",
              "aqua": "#1baf7a", "neutral": "#8c959f"},
    "dark": {"surface": "#0d1117", "text": "#f0f6fc", "text2": "#9198a1", "muted": "#6e7681",
             "grid": "#21262d", "axis": "#3d444d", "blue": "#3987e5", "orange": "#d95926",
             "aqua": "#199e70", "neutral": "#6e7681"},
}


class Svg:
    def __init__(self, w: int, h: int, theme: dict, title: str):
        self.w, self.h, self.t = w, h, theme
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                      f'role="img" aria-label="{escape(title)}" font-family="{FONT}">',
                      f"<title>{escape(title)}</title>"]

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=12, color="text", weight=400, anchor="start", tabular=False) -> None:
        style = ' style="font-variant-numeric:tabular-nums"' if tabular else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{self.t[color]}" '
                 f'text-anchor="{anchor}"{style}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, color="grid", width=1, dash=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{self.t[color]}" '
                 f'stroke-width="{width}"{d}/>')

    def hbar(self, x0, y, w, h, color, r=4) -> None:
        """Horizontal bar: square at the baseline, 4px rounded data end."""
        r = min(r, w / 2, h / 2)
        self.add(f'<path d="M{x0:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
                 f'a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z" fill="{self.t[color]}"/>')

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.parts + ["</svg>"]) + "\n")


def nice_ticks(vmax: float, n: int = 4) -> list[float]:
    raw = vmax / n
    p = 10 ** math.floor(math.log10(raw))
    step = next(m * p for m in (1, 2, 2.5, 5, 10) if m * p >= raw)
    return [i * step for i in range(int(math.ceil(vmax / step - 1e-9)) + 1)]


def compact(v: float) -> str:
    return f"{v / 1000:g}K" if v >= 1000 else f"{v:g}"


def legend_key(s: Svg, x, y, color, label, kind="bar", dash=None) -> float:
    if kind == "bar":
        s.add(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{s.t[color]}"/>')
        tx = x + 15
    else:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s.add(f'<line x1="{x}" y1="{y - 4}" x2="{x + 22}" y2="{y - 4}" stroke="{s.t[color]}" stroke-width="2" '
              f'stroke-linecap="round"{d}/>')
        tx = x + 28
    s.text(tx, y, label, 12, "text2")
    return tx + 6.6 * len(label) + 18


def hbar_chart(path, theme, title, subtitle, rows, unit, note, extra_col=None, width=430, min_rows=0):
    """rows: [(label, value, color_key, value_label)]; min_rows pads the height so side-by-side charts align."""
    t = THEMES[theme]
    bar_h, gap, top = 18, 12, 92
    label_w = 150
    right = 70 + (52 if extra_col else 0)
    h = top + max(len(rows), min_rows) * (bar_h + gap) + 44
    s = Svg(width, h, t, title)
    s.text(0, 18, title, 15, "text", 600)
    s.text(0, 37, subtitle, 12, "text2")
    lx = legend_key(s, 0, 62, "blue", "Redline")
    legend_key(s, lx, 62, "neutral", "Baselines")
    x0, x1 = label_w, width - right
    vmax = max(r[1] for r in rows)
    ticks = nice_ticks(vmax)
    sx = lambda v: x0 + v / ticks[-1] * (x1 - x0)  # noqa: E731
    y_end = top + len(rows) * (bar_h + gap) - gap + 6
    for v in ticks:
        s.line(sx(v), top - 6, sx(v), y_end, "axis" if v == 0 else "grid")
        s.text(sx(v), y_end + 16, compact(v), 11, "muted", anchor="middle", tabular=True)
    s.text(x1, y_end + 32, unit, 11, "muted", anchor="end")
    if extra_col:
        s.text(width, top - 12, extra_col[0], 11, "muted", anchor="end")
    for i, (label, v, color, vl) in enumerate(rows):
        y = top + i * (bar_h + gap)
        s.text(label_w - 10, y + bar_h / 2 + 4, label, 12.5, "text", anchor="end")
        s.hbar(x0, y, sx(v) - x0, bar_h, color)
        s.text(sx(v) + 6, y + bar_h / 2 + 4, vl, 12, "text", 600, tabular=True)
        if extra_col:
            s.text(width, y + bar_h / 2 + 4, extra_col[1][i], 12, "text2", anchor="end", tabular=True)
    if note:
        s.text(0, h - 4, note, 11, "muted")
    s.save(path)


def sweep_chart(path, theme, sweep):
    t = THEMES[theme]
    W, H = 860, 400
    s = Svg(W, H, t, "Decode throughput vs batch size")
    s.text(0, 18, "Decode throughput vs batch size", 15, "text", 600)
    s.text(0, 37, "Llama 3.1 8B on one RTX 4090 · 512-token prompts, 256 output tokens · tokens/s across the batch "
                  "(higher is better)", 12, "text2")
    series = [  # tag, label, color, dash
        ("redline_int4", "Redline INT4", "orange", None),
        ("redline_int4_gemv", "Redline INT4, GEMV only", "muted", "1 4"),
        ("redline_int8", "Redline INT8", "aqua", None),
        ("redline_fp16", "Redline FP16", "blue", None),
        ("vllm_awq_int4", "vLLM AWQ INT4", "orange", "6 4"),
        ("vllm", "vLLM FP16", "blue", "6 4"),
        ("hf_fp16", "HF Transformers FP16", "neutral", "2 4"),
    ]
    series = [x for x in series if x[0] in sweep]
    lx, ly = 0, 64
    for tag, label, color, dash in series:
        if lx > W - 190:  # wrap the legend
            lx, ly = 0, ly + 20
        lx = legend_key(s, lx, ly, color, label, "line", dash)
    batches = [1, 2, 4, 8, 16, 32]
    data = {tag: [report.decode_tps(sweep[tag][f"sweep_b{b}_in512_out256"]) for b in batches]
            for tag, *_ in series}
    l, r, top, bottom = 52, 150, ly + 24, H - 42
    ticks = nice_ticks(max(max(v) for v in data.values()))
    sx = lambda i: l + i / (len(batches) - 1) * (W - l - r)  # noqa: E731
    sy = lambda v: bottom - v / ticks[-1] * (bottom - top)  # noqa: E731
    for v in ticks:
        s.line(l, sy(v), W - r, sy(v), "axis" if v == 0 else "grid")
        s.text(l - 8, sy(v) + 4, compact(v), 11, "muted", anchor="end", tabular=True)
    for i, b in enumerate(batches):
        s.text(sx(i), bottom + 18, str(b), 11, "muted", anchor="middle", tabular=True)
    s.text(W - r, bottom + 34, "batch size (concurrent sequences)", 11, "muted", anchor="end")
    for tag, label, color, dash in reversed(series):  # Redline drawn last, on top
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(data[tag]))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s.add(f'<polyline points="{pts}" fill="none" stroke="{t[color]}" stroke-width="2" stroke-linejoin="round" '
              f'stroke-linecap="round"{d}/>')
        if not dash:
            for i, v in enumerate(data[tag]):
                s.add(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4" fill="{t[color]}" stroke="{t["surface"]}" '
                      f'stroke-width="2"/>')
    # Selective end labels: the two INT4 systems, nudged apart when their batch-32 values are close.
    xe = sx(len(batches) - 1) + 10
    va, vr = data["vllm_awq_int4"][-1], data["redline_int4"][-1]
    ya, yr = sy(va), sy(vr)
    if abs(ya - yr) < 40:
        mid = (ya + yr) / 2
        ya, yr = (mid - 20, mid + 20) if va >= vr else (mid + 20, mid - 20)
    s.text(xe, ya + 4, f"vLLM AWQ INT4 {va:,.0f}", 12, "text", 600, tabular=True)
    s.text(xe, ya + 19 if ya < yr else ya - 12, "Marlin tensor cores", 11.5, "text2")
    s.text(xe, yr + 4, f"Redline INT4 {vr:,.0f}", 12, "text", 600, tabular=True)
    s.text(xe, yr + 19 if yr < ya else yr - 12, "tensor-core W4A16", 11.5, "text2")
    # Where Redline INT4 leads every system (and by how much over vLLM AWQ): the upper-left of the
    # plot is empty, so the note sits there with a leader line down to the batch-2 point.
    others = [tag for tag, *_ in series if not tag.startswith("redline_int4")]
    lead = [i for i in range(len(batches)) if data["redline_int4"][i] >= max(data[o][i] for o in others)]
    if lead and lead[0] == 0:
        n = 0
        while n + 1 < len(lead) and lead[n + 1] == n + 1:
            n += 1
        gains = [data["redline_int4"][i] / data["vllm_awq_int4"][i] - 1 for i in range(n + 1)]
        i = min(1, n)
        yt = sy(ticks[-1] * 0.78)
        s.line(sx(i), yt + 8, sx(i), sy(data["redline_int4"][i]) - 8, "axis")
        span = f"Batch 1–{batches[n]}" if n else "Batch 1"
        s.text(sx(0) + 12, yt - 14, f"{span}: Redline INT4 is the fastest system", 12, "text", 600)
        lo, hi = round(min(gains) * 100), round(max(gains) * 100)
        s.text(sx(0) + 12, yt + 1, f"{lo}–{hi}% ahead of vLLM's Marlin kernel" if lo != hi
               else f"{lo}% ahead of vLLM's Marlin kernel", 12, "text2")
    s.save(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "rtx4090"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "img"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    quick = report.load_reports(os.path.join(args.results, "quick"))
    sweep = report.load_reports(os.path.join(args.results, "sweep"))

    w1 = "offline_b1_in128_out128_greedy"
    acc = quick["redline_int4_spec"][w1]["speculative"]["acceptance_rate"]
    b1 = [("Redline INT4 + spec", "redline_int4_spec"), ("Redline INT4", "redline_int4"),
          ("vLLM AWQ INT4", "vllm_awq_int4"), ("Redline INT8", "redline_int8"), ("Redline FP16", "redline_fp16"),
          ("vLLM FP16", "vllm"), ("HF Transformers FP16", "hf_fp16")]
    rows1 = []
    for label, tag in sorted(b1, key=lambda x: -report.decode_tps(quick[x[1]][w1])):
        v = report.decode_tps(quick[tag][w1])
        rows1.append((label, v, "blue" if tag.startswith("redline") else "neutral", f"{v:.0f}"))

    w8 = "offline_b8_in1024_out128_greedy"
    mem = [("HF Transformers FP16", "hf_fp16"), ("Redline FP16", "redline_fp16"), ("Redline INT8", "redline_int8"),
           ("Redline INT8 + INT8 KV", "redline_int8_kvint8"), ("Redline INT4", "redline_int4")]
    import json
    with open(os.path.join(args.results, "quality.json")) as f:
        q = json.load(f)
    ret = {os.path.basename(c["model"]): c["retention"]["mean"] for c in q["candidates"]}
    # INT4 quality is the AWQ + GPTQ calibrated file: same format and size as round-to-nearest.
    q4 = ret.get("l8b-int4-awq_gptq.rlm", ret["l8b-int4.rlm"])
    quality = {"hf_fp16": "reference", "redline_int8": f"{ret['l8b-int8.rlm'] * 100:.1f}%",
               "redline_int4": f"{q4 * 100:.1f}%"}
    q4_note = (f" INT4: AWQ + GPTQ ({ret['l8b-int4.rlm'] * 100:.1f}% without calibration)."
               if "l8b-int4-awq_gptq.rlm" in ret else "")
    rows3 = [(label, report.peak_gib(quick[tag][w8]), "blue" if tag.startswith("redline") else "neutral",
              f"{report.peak_gib(quick[tag][w8]):.1f}") for label, tag in mem]
    col3 = ("quality", [quality.get(tag, "–") for _, tag in mem])

    for theme in THEMES:
        hbar_chart(os.path.join(args.out, f"decode_batch1-{theme}.svg"), theme, "Decode speed, one request",
                   "Llama 3.1 8B · RTX 4090 · tokens/s (higher is better)", rows1, "tokens / s",
                   f"WikiText prompts, 128 in / 128 out. Speculative: 1B draft, {acc * 100:.0f}% accepted.",
                   min_rows=len(rows1))
        hbar_chart(os.path.join(args.out, f"memory_quality-{theme}.svg"), theme, "Peak GPU memory and quality",
                   "Batch 8, 1024-token prompts · GiB (lower is better)", rows3, "GiB",
                   "Quality: mean retention on WikiText-2, HellaSwag, ARC-E/C." + q4_note, extra_col=col3,
                   min_rows=len(rows1))
        sweep_chart(os.path.join(args.out, f"batch_sweep-{theme}.svg"), theme, sweep)
    print(f"wrote {len(os.listdir(args.out))} files to {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
