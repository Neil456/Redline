"""Evaluation-quality retention of Redline builds vs the FP16 HF model.

Both systems are scored with identical teacher-forced log-likelihoods:
  * wikitext   - WikiText-2 test perplexity over non-overlapping windows
  * hellaswag  - 4-way completion, length-normalized accuracy (acc_norm)
  * arc_easy / arc_challenge - multiple choice accuracy (acc / acc_norm)
Retention = metric(redline) / metric(hf), with perplexity inverted
(ppl_hf / ppl_redline). The overall score is the mean retention across tasks.

    python -m redline.eval_quality --hf meta-llama/Llama-3.1-8B-Instruct \\
        --redline llama31-8b-int8.rlm --redline llama31-8b-int4.rlm --limit 1000 --out quality.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

TASKS = ("wikitext", "hellaswag", "arc_easy", "arc_challenge")


class HFScorer:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda").eval()

    def score(self, tokens: list[int]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            ids = torch.tensor([tokens], device="cuda")
            logits = self.model(input_ids=ids).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1).gather(1, ids[0, 1:, None])[:, 0]
        return lp.cpu().numpy()


class RedlineScorer:
    def __init__(self, path: str, max_seq_len: int):
        from .runtime import load_engine

        self.engine = load_engine(path, max_seq_len=max_seq_len, max_seqs=1, max_batched_tokens=max_seq_len,
                                  max_logit_rows=512, kv_cache_tokens=max_seq_len + 64, cuda_graphs=False)

    def score(self, tokens: list[int]) -> np.ndarray:
        return np.asarray(self.engine.score(tokens), dtype=np.float64)


# ---------------------------------------------------------------- task data


def wikitext_windows(tok, window: int, limit: int) -> list[list[int]]:
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    out = []
    for i in range(0, len(ids) - window, window):
        out.append(bos + ids[i : i + window])
        if len(out) >= limit:
            break
    return out


def mc_examples(task: str, limit: int) -> list[dict]:
    """Returns [{"context": str, "choices": [str], "label": int}] (lm-eval-harness prompts)."""
    from datasets import load_dataset

    if task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")

        def prep(t):
            return t.strip().replace(" [title]", ". ").replace("  ", " ")

        ex = [{"context": prep(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()),
               "choices": [" " + prep(e) for e in d["endings"]], "label": int(d["label"])} for d in ds]
    elif task in ("arc_easy", "arc_challenge"):
        name = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", name, split="test")
        ex = [{"context": f"Question: {d['question']}\nAnswer:",
               "choices": [" " + c for c in d["choices"]["text"]],
               "label": d["choices"]["label"].index(d["answerKey"])} for d in ds]
    else:
        raise ValueError(task)
    return ex[:limit]


# ---------------------------------------------------------------- evaluation


def eval_ppl(scorer, windows) -> dict:
    nll, n = 0.0, 0
    for w in windows:
        lp = scorer.score(w)
        nll -= float(lp.sum())
        n += len(lp)
    return {"perplexity": math.exp(nll / n), "tokens": n}


def eval_mc(scorer, tok, examples, normalize: bool) -> dict:
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    correct = correct_norm = 0
    for ex in examples:
        ctx = bos + tok(ex["context"], add_special_tokens=False)["input_ids"]
        scores, norm = [], []
        for choice in ex["choices"]:
            cont = tok(choice, add_special_tokens=False)["input_ids"]
            lp = scorer.score(ctx + cont)
            s = float(lp[len(ctx) - 1 : len(ctx) - 1 + len(cont)].sum())
            scores.append(s)
            norm.append(s / max(1, len(choice)))
        correct += int(np.argmax(scores) == ex["label"])
        correct_norm += int(np.argmax(norm) == ex["label"])
    n = len(examples)
    return {"acc": correct / n, "acc_norm": correct_norm / n, "n": n, "primary": "acc_norm" if normalize else "acc"}


def evaluate(scorer, tok, tasks, limit: int, window: int) -> dict:
    res = {}
    for task in tasks:
        t0 = time.time()
        if task == "wikitext":
            res[task] = eval_ppl(scorer, wikitext_windows(tok, window, limit))
        else:
            res[task] = eval_mc(scorer, tok, mc_examples(task, limit), normalize=task != "arc_easy")
        res[task]["seconds"] = time.time() - t0
        print(f"  {task}: {res[task]}", file=sys.stderr)
    return res


def primary(task: str, r: dict) -> float:
    return r["perplexity"] if task == "wikitext" else r[r["primary"]]


def retention(hf: dict, rl: dict) -> dict:
    per = {}
    for task in hf:
        a, b = primary(task, hf[task]), primary(task, rl[task])
        per[task] = a / b if task == "wikitext" else b / a
    return {"per_task": per, "mean": float(np.mean(list(per.values())))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True, help="reference HF model id (FP16)")
    ap.add_argument("--redline", action="append", required=True, help=".rlm file(s) to evaluate")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=1000, help="examples (or ppl windows) per task")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--hf-cache", help="reuse/store HF reference results in this JSON file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf)
    tasks = [t for t in args.tasks.split(",") if t]
    hf_res = None
    if args.hf_cache:
        try:
            with open(args.hf_cache) as f:
                cached = json.load(f)
            if cached.get("limit") == args.limit and all(t in cached["results"] for t in tasks):
                hf_res = {t: cached["results"][t] for t in tasks}
        except FileNotFoundError:
            pass
    if hf_res is None:
        print(f"evaluating HF {args.hf}", file=sys.stderr)
        scorer = HFScorer(args.hf)
        hf_res = evaluate(scorer, tok, tasks, args.limit, args.window)
        del scorer
        import torch

        torch.cuda.empty_cache()
        if args.hf_cache:
            with open(args.hf_cache, "w") as f:
                json.dump({"model": args.hf, "limit": args.limit, "results": hf_res}, f, indent=2)
    report = {"reference": {"model": args.hf, "results": hf_res}, "candidates": []}
    for path in args.redline:
        print(f"evaluating Redline {path}", file=sys.stderr)
        rl = evaluate(RedlineScorer(path, args.window + 16 + 512), tok, tasks, args.limit, args.window)
        ret = retention(hf_res, rl)
        report["candidates"].append({"model": path, "results": rl, "retention": ret})
        print(f"  retention {ret['mean'] * 100:.2f}% {ret['per_task']}", file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
