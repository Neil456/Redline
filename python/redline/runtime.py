"""Python access to the Redline engine.

Uses the pybind11 extension (``redline._redline``, built with
``-DREDLINE_BUILD_PYTHON=ON``) when available, otherwise shells out to the
``redline_generate`` binary for generation and scoring.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from typing import Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_binary(name: str) -> str:
    env = os.environ.get("REDLINE_BUILD_DIR")
    candidates = [os.path.join(env, name)] if env else []
    candidates += sorted(glob.glob(os.path.join(ROOT, "build*", name)))
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found; build Redline (cmake -B build && cmake --build build) "
                            "or set REDLINE_BUILD_DIR")


def _ext():
    try:
        from . import _redline  # type: ignore

        return _redline
    except ImportError:
        return None


class Engine:
    """In-process engine (requires the pybind11 extension).

    Options mirror the CLI flags: max_seqs, max_batched_tokens, max_seq_len,
    kv_cache_tokens, gpu_memory_utilization, kv_dtype, cuda_graphs,
    async_scheduling, int8_gemm, int4_gemm, draft_model, spec_tokens, ...
    """

    def __init__(self, model: str, **options):
        ext = _ext()
        if ext is None:
            raise ImportError(
                "redline._redline is not built; configure with -DREDLINE_BUILD_PYTHON=ON "
                "or use redline.runtime.CliEngine"
            )
        cfg = {"model": model, **{k: v for k, v in options.items() if v is not None}}
        self._engine = ext.Engine(cfg)

    @property
    def vocab_size(self) -> int:
        return self._engine.vocab_size

    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        ignore_eos: bool = False,
        stop_token_ids: Sequence[int] = (),
    ) -> list[dict]:
        return self._engine.generate(
            [list(p) for p in prompts], max_new_tokens, temperature, top_k, top_p, seed, ignore_eos,
            list(stop_token_ids),
        )

    def score(self, tokens: Sequence[int]) -> list[float]:
        """log p(tokens[i] | tokens[:i]) for i = 1 .. len(tokens) - 1."""
        return self._engine.score(list(tokens))

    def benchmark(self, workload: dict, online: bool = False, warmup: int = 2) -> dict:
        return json.loads(self._engine.benchmark(json.dumps(workload), online, warmup))

    def memory_report(self) -> dict:
        return json.loads(self._engine.memory_report())

    def describe(self) -> dict:
        return json.loads(self._engine.describe())


class CliEngine:
    """Subprocess fallback around ``redline_generate`` (one engine per call)."""

    def __init__(self, model: str, **options):
        self.model = model
        negated = {"cuda_graphs": "--no-cuda-graphs", "async_scheduling": "--no-async",
                   "mix_prefill_decode": "--no-mixed-batches"}
        self.flags: list[str] = []
        for k, v in options.items():
            if v is None:
                continue
            if k in negated:
                if v is False:
                    self.flags.append(negated[k])
                continue
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    self.flags.append(flag)
                continue
            self.flags += [flag, str(v)]
        self.binary = find_binary("redline_generate")

    def _run(self, extra: list[str]) -> list[dict]:
        out = subprocess.run([self.binary, "--model", self.model, *self.flags, *extra],
                             check=True, capture_output=True, text=True)
        return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]

    def generate(self, prompts, max_new_tokens=128, temperature=0.0, top_k=0, top_p=1.0, seed=0,
                 ignore_eos=False, stop_token_ids=()) -> list[dict]:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([list(p) for p in prompts], f)
            path = f.name
        try:
            args = ["--prompts-json", path, "--max-new-tokens", str(max_new_tokens), "--temperature",
                    str(temperature), "--top-k", str(top_k), "--top-p", str(top_p), "--seed", str(seed)]
            if ignore_eos:
                args.append("--ignore-eos")
            return self._run(args)
        finally:
            os.unlink(path)

    def score(self, tokens: Sequence[int]) -> list[float]:
        return self._run(["--score-ids", ",".join(map(str, tokens))])[0]["logprobs"]


def load_engine(model: str, **options):
    """Engine if the extension is available, else CliEngine."""
    return Engine(model, **options) if _ext() is not None else CliEngine(model, **options)
