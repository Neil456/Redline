"""Reproducible benchmark workloads (JSON) shared by Redline, HF and vLLM runs.

    # one fixed-shape workload
    python -m redline.workloads synthetic --out w.json --num-requests 16 --prompt-len 512 --output-len 256

    # prompts from a text corpus, Poisson arrivals at 4 req/s
    python -m redline.workloads corpus --tokenizer meta-llama/Llama-3.1-8B-Instruct \\
        --num-requests 200 --rate 4 --out online.json

    # the full grid (hundreds of workloads) + manifest.json
    python -m redline.workloads suite --tokenizer meta-llama/Llama-3.1-8B-Instruct --out-dir workloads/

Workload schema: {"name", "description", "seed", "settings": {...}, "requests": [
  {"id", "prompt_ids" | "prompt_len", "max_new_tokens", "temperature", "top_k", "top_p",
   "seed", "arrival_time", "ignore_eos"}]}
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from dataclasses import dataclass

SAMPLING = {
    "greedy": {"temperature": 0.0, "top_k": 0, "top_p": 1.0},
    "nucleus": {"temperature": 0.8, "top_k": 0, "top_p": 0.95},
    "topk": {"temperature": 0.7, "top_k": 50, "top_p": 1.0},
}


class Corpus:
    """Source of real-text token ids for prompts (falls back to random ids)."""

    def __init__(self, tokenizer: str | None, source: str = "wikitext", seed: int = 0, vocab_size: int = 32000):
        self.rng = random.Random(seed)
        self.vocab_size = vocab_size
        self.ids: list[int] = []
        self.bos: int | None = None
        if tokenizer is None or source == "random":
            return
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer)
        self.vocab_size = len(tok)
        self.bos = tok.bos_token_id
        text = self._load_text(source)
        self.ids = tok(text, add_special_tokens=False)["input_ids"]

    @staticmethod
    def _load_text(source: str) -> str:
        if source.startswith("file:"):
            with open(source[5:], encoding="utf-8") as f:
                return f.read()
        if source == "wikitext":
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            return "\n".join(t for t in ds["text"] if t.strip())[:4_000_000]
        raise ValueError(f"unknown corpus source {source}")

    def prompt(self, length: int) -> list[int]:
        if not self.ids:
            lo = min(1000, self.vocab_size // 4)
            return [self.rng.randrange(lo, self.vocab_size) for _ in range(length)]
        body = length - (1 if self.bos is not None else 0)
        start = self.rng.randrange(0, max(1, len(self.ids) - body))
        ids = self.ids[start : start + body]
        return ([self.bos] if self.bos is not None else []) + ids


def make_requests(corpus: Corpus, n: int, prompt_len, output_len, sampling: str, seed: int,
                  rate: float | None = None) -> list[dict]:
    rng = random.Random(seed)
    t = 0.0
    reqs = []
    for i in range(n):
        pl = prompt_len() if callable(prompt_len) else prompt_len
        ol = output_len() if callable(output_len) else output_len
        if rate:
            t += rng.expovariate(rate)
        reqs.append({
            "id": i,
            "prompt_ids": corpus.prompt(pl),
            "max_new_tokens": ol,
            **SAMPLING[sampling],
            "seed": seed * 100003 + i,
            "arrival_time": round(t, 6) if rate else 0.0,
            "ignore_eos": True,
        })
    return reqs


def workload(name: str, description: str, settings: dict, requests: list[dict], seed: int) -> dict:
    return {"name": name, "description": description, "seed": seed, "settings": settings, "requests": requests}


@dataclass
class SuiteSpec:
    prompt_lens: tuple = (128, 512, 1024, 2048)
    output_lens: tuple = (128, 256, 512)
    batch_sizes: tuple = (1, 4, 8, 16, 32)
    samplings: tuple = ("greedy", "nucleus")
    batches_per_workload: int = 2
    online_rates: tuple = (1.0, 2.0, 4.0, 8.0, 16.0)
    online_requests: int = 200


QUICK = SuiteSpec(prompt_lens=(128, 1024), output_lens=(128,), batch_sizes=(1, 8), samplings=("greedy",),
                  online_rates=(4.0,), online_requests=50)


def build_suite(corpus: Corpus, out_dir: str, spec: SuiteSpec, seed: int = 0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i, (pl, ol, bs, smp) in enumerate(itertools.product(spec.prompt_lens, spec.output_lens, spec.batch_sizes,
                                                           spec.samplings)):
        name = f"offline_b{bs}_in{pl}_out{ol}_{smp}"
        s = {"kind": "offline", "batch_size": bs, "prompt_len": pl, "output_len": ol, "sampling": smp}
        reqs = make_requests(corpus, bs * spec.batches_per_workload, pl, ol, smp, seed + i)
        w = workload(name, f"static batch {bs}, {pl} prompt / {ol} output tokens, {smp}", s, reqs, seed + i)
        path = os.path.join(out_dir, name + ".json")
        with open(path, "w") as f:
            json.dump(w, f)
        entries.append({"name": name, "path": os.path.basename(path), "settings": s})
    for j, (rate, smp) in enumerate(itertools.product(spec.online_rates, spec.samplings)):
        name = f"online_rate{rate:g}_{smp}"
        rng = random.Random(seed + 7919 * (j + 1))
        # ShareGPT-like length mix: log-uniform prompts, variable outputs.
        pl = lambda: int(min(2048, max(16, round(2 ** rng.uniform(5, 11)))))  # noqa: E731
        ol = lambda: int(rng.choice([64, 128, 256, 384, 512]))  # noqa: E731
        s = {"kind": "online", "rate": rate, "sampling": smp, "batch_size": 32}
        reqs = make_requests(corpus, spec.online_requests, pl, ol, smp, seed + 10000 + j, rate=rate)
        w = workload(name, f"Poisson arrivals at {rate:g} req/s, mixed lengths, {smp}", s, reqs, seed + 10000 + j)
        path = os.path.join(out_dir, name + ".json")
        with open(path, "w") as f:
            json.dump(w, f)
        entries.append({"name": name, "path": os.path.basename(path), "settings": s})
    manifest = {"seed": seed, "num_workloads": len(entries), "workloads": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("synthetic", "corpus", "suite"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--sampling", default="greedy", choices=list(SAMPLING))
        if name != "synthetic":
            p.add_argument("--tokenizer", required=True)
            p.add_argument("--source", default="wikitext", help="wikitext | file:PATH | random")
        else:
            p.add_argument("--vocab-size", type=int, default=32000)
        if name == "suite":
            p.add_argument("--out-dir", required=True)
            p.add_argument("--quick", action="store_true", help="small grid for smoke tests")
        else:
            p.add_argument("--out", required=True)
            p.add_argument("--num-requests", type=int, default=16)
            p.add_argument("--prompt-len", type=int, default=512)
            p.add_argument("--output-len", type=int, default=128)
            p.add_argument("--rate", type=float, default=None, help="Poisson arrival rate (req/s)")
    args = ap.parse_args(argv)

    if args.cmd == "suite":
        corpus = Corpus(args.tokenizer, args.source, args.seed)
        m = build_suite(corpus, args.out_dir, QUICK if args.quick else SuiteSpec(), args.seed)
        print(f"wrote {m['num_workloads']} workloads to {args.out_dir}")
        return
    corpus = Corpus(None if args.cmd == "synthetic" else args.tokenizer,
                    "random" if args.cmd == "synthetic" else args.source, args.seed,
                    vocab_size=getattr(args, "vocab_size", 32000))
    reqs = make_requests(corpus, args.num_requests, args.prompt_len, args.output_len, args.sampling, args.seed,
                         rate=args.rate)
    s = {"kind": "online" if args.rate else "offline", "batch_size": args.num_requests,
         "prompt_len": args.prompt_len, "output_len": args.output_len, "sampling": args.sampling}
    name = os.path.splitext(os.path.basename(args.out))[0]
    with open(args.out, "w") as f:
        json.dump(workload(name, "", s, reqs, args.seed), f)
    print(f"wrote {args.out} ({len(reqs)} requests)")


if __name__ == "__main__":
    main()
