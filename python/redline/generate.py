"""Text generation with Redline (tokenization via the HF tokenizer).

    python -m redline.generate --model llama31-8b-int4.rlm \\
        --tokenizer meta-llama/Llama-3.1-8B-Instruct --prompt "The capital of France is" \\
        --max-new-tokens 64 --chat
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .format import RedlineFile
from .runtime import load_engine


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", help="HF tokenizer id/path (default: model's source checkpoint)")
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draft-model")
    ap.add_argument("--spec-tokens", type=int)
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--json", action="store_true", help="print raw JSON results")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    source = args.tokenizer or RedlineFile(args.model).spec.source
    if not source:
        sys.exit("--tokenizer is required (model file has no source checkpoint recorded)")
    tok = AutoTokenizer.from_pretrained(source)
    prompts = []
    for p in args.prompt:
        if args.chat:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        else:
            ids = tok(p)["input_ids"]
        prompts.append(list(ids))

    engine = load_engine(
        args.model, draft_model=args.draft_model, spec_tokens=args.spec_tokens, kv_dtype=args.kv_dtype,
        max_seq_len=args.max_seq_len, gpu_memory_utilization=0.5, max_seqs=max(1, len(prompts)),
    )
    t0 = time.perf_counter()
    outs = engine.generate(prompts, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                           top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    dt = time.perf_counter() - t0
    for o in outs:
        text = tok.decode(o["output_ids"], skip_special_tokens=True)
        if args.json:
            print(json.dumps({**o, "text": text}))
        else:
            print(f"--- [{o['id']}] ({len(o['output_ids'])} tokens, {o.get('finish_reason')})")
            print(args.prompt[o["id"]] + text)
    n = sum(len(o["output_ids"]) for o in outs)
    print(f"\n{n} tokens in {dt:.2f}s ({n / dt:.1f} tok/s incl. startup)", file=sys.stderr)


if __name__ == "__main__":
    main()
