"""Benchmark metrics shared by the Python baselines; mirrors the report schema
of the C++ harness (src/cuda/bench/harness.cpp) so results are comparable."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


def summarize(values) -> dict:
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0,
                "std": 0.0}
    return {
        "count": int(v.size),
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "min": float(v.min()),
        "max": float(v.max()),
        "std": float(v.std()),
    }


@dataclass
class RequestTrace:
    id: int
    prompt_len: int
    arrival: float
    token_times: list = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return len(self.token_times)


class DeviceMemorySampler:
    """Samples device-wide used memory (NVML if available, else torch) in a
    background thread, matching Redline's cudaMemGetInfo-based peak."""

    def __init__(self, device: int = 0, interval_s: float = 0.005):
        self.device = device
        self.interval = interval_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception:  # noqa: BLE001
            self._nvml = None

    def used(self) -> int:
        if self._nvml is not None:
            return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        import torch

        free, total = torch.cuda.mem_get_info(self.device)
        return int(total - free)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.used()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.used())


def build_report(system: str, workload: dict, traces: list[RequestTrace], duration_s: float,
                 decode_tokens: int, decode_time_s: float, peak_bytes: int, extra: dict | None = None) -> dict:
    ttft, itl, tpot, e2e = [], [], [], []
    out_tokens = 0
    for t in traces:
        if not t.token_times:
            continue
        out_tokens += t.num_tokens
        ttft.append((t.token_times[0] - t.arrival) * 1000)
        e2e.append((t.token_times[-1] - t.arrival) * 1000)
        if t.num_tokens > 1:
            tpot.append((t.token_times[-1] - t.token_times[0]) * 1000 / (t.num_tokens - 1))
        itl.extend((b - a) * 1000 for a, b in zip(t.token_times, t.token_times[1:]))
    report = {
        "system": system,
        "workload": workload["name"],
        "workload_settings": workload.get("settings", {}),
        "num_requests": len(traces),
        "duration_s": duration_s,
        "prompt_tokens": sum(t.prompt_len for t in traces),
        "output_tokens": out_tokens,
        "request_throughput_rps": len(traces) / duration_s,
        "output_throughput_tok_s": out_tokens / duration_s,
        "decode_throughput_tok_s": decode_tokens / decode_time_s if decode_time_s > 0 else 0.0,
        "ttft_ms": summarize(ttft),
        "itl_ms": summarize(itl),
        "tpot_ms": summarize(tpot),
        "e2e_latency_ms": summarize(e2e),
        "memory": {"peak_device_used_bytes": int(peak_bytes)},
        "speculative": {"enabled": False, "acceptance_rate": 0.0},
    }
    if extra:
        report.update(extra)
    return report


def decode_window(traces: list[RequestTrace]) -> tuple[int, float]:
    """Tokens and wall time of the decode phase, measured per batch as the
    span after every request in it has its first token (TTFT excluded)."""
    if not traces:
        return 0, 0.0
    start = max(t.token_times[0] for t in traces if t.token_times)
    end = max(t.token_times[-1] for t in traces if t.token_times)
    tokens = sum(sum(1 for x in t.token_times if x > start) for t in traces)
    return tokens, end - start
