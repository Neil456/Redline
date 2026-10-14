"""Tests for the workload generator, metrics, report aggregation and suite runner."""

from __future__ import annotations

import json
import os

import numpy as np

from redline import metrics, report, suite, workloads
from redline.runtime import CliEngine


def test_suite_is_reproducible_and_complete(tmp_path):
    corpus = workloads.Corpus(None, "random", seed=3, vocab_size=1000)
    m1 = workloads.build_suite(corpus, str(tmp_path / "a"), workloads.QUICK, seed=3)
    corpus = workloads.Corpus(None, "random", seed=3, vocab_size=1000)
    m2 = workloads.build_suite(corpus, str(tmp_path / "b"), workloads.QUICK, seed=3)
    q = workloads.QUICK
    expected = len(q.prompt_lens) * len(q.output_lens) * len(q.batch_sizes) * len(q.samplings) + len(
        q.online_rates) * len(q.samplings)
    assert m1["num_workloads"] == expected == m2["num_workloads"]
    for e in m1["workloads"]:
        a = json.load(open(tmp_path / "a" / e["path"]))
        b = json.load(open(tmp_path / "b" / e["path"]))
        assert a == b  # same seed -> identical workload
        s = e["settings"]
        for r in a["requests"]:
            assert all(0 <= t < 1000 for t in r["prompt_ids"])
            if s["kind"] == "offline":
                assert len(r["prompt_ids"]) == s["prompt_len"]
                assert r["max_new_tokens"] == s["output_len"]
                assert r["arrival_time"] == 0.0
        if s["kind"] == "online":
            times = [r["arrival_time"] for r in a["requests"]]
            assert times == sorted(times) and times[-1] > 0
    full = workloads.SuiteSpec()
    n_full = len(full.prompt_lens) * len(full.output_lens) * len(full.batch_sizes) * len(full.samplings)
    assert n_full + len(full.online_rates) * len(full.samplings) >= 100


def test_metrics_summary_and_report():
    s = metrics.summarize(range(1, 101))
    assert s["count"] == 100 and abs(s["p50"] - 50.5) < 1e-9 and abs(s["p99"] - 99.01) < 1e-9
    traces = []
    for i in range(2):
        t = metrics.RequestTrace(i, 10, arrival=0.0)
        t.token_times = [0.1 + i * 0.01, 0.2, 0.3, 0.4]
        traces.append(t)
    n, dt = metrics.decode_window(traces)
    assert n == 6 and abs(dt - 0.29) < 1e-9
    w = {"name": "w", "settings": {"kind": "offline", "batch_size": 2}}
    rep = metrics.build_report("x", w, traces, duration_s=0.5, decode_tokens=n, decode_time_s=dt, peak_bytes=2**30)
    assert rep["output_tokens"] == 8
    assert abs(rep["ttft_ms"]["p50"] - 105.0) < 1e-6
    assert rep["itl_ms"]["count"] == 6
    assert rep["memory"]["peak_device_used_bytes"] == 2**30


def _fake(tag, name, itl_p50, peak_gib, batch=8):
    return {"tag": tag, "workload": name, "workload_settings": {"kind": "offline", "batch_size": batch},
            "ttft_ms": metrics.summarize([10.0]), "itl_ms": {**metrics.summarize([itl_p50]), "p50": itl_p50},
            "memory": {"peak_device_used_bytes": int(peak_gib * 2**30)}, "decode_throughput_tok_s": 0.0}


def test_report_comparison(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    for name, (rl, hf) in {"w1": (10.0, 20.0), "w2": (5.0, 20.0)}.items():
        json.dump(_fake("redline_int4", name, rl, 6.0), open(d / f"redline__{name}.json", "w"))
        json.dump(_fake("hf_fp16", name, hf, 16.0), open(d / f"hf__{name}.json", "w"))
    reps = report.load_reports(str(d))
    c = report.compare(reps, "redline_int4", "hf_fp16")
    assert c["workloads"] == 2
    assert abs(c["decode_speedup_geomean"] - np.sqrt(2.0 * 4.0)) < 1e-9
    assert abs(c["peak_vram_reduction_mean"] - (1 - 6 / 16)) < 1e-9
    out = tmp_path / "r.md"
    report.main(["--results", str(d), "--baseline", "hf_fp16", "--out", str(out)])
    text = out.read_text()
    assert "redline_int4" in text and "+182.8%" in text


def test_suite_dry_run_groups_by_batch(tmp_path, capsys):
    corpus = workloads.Corpus(None, "random", seed=0, vocab_size=1000)
    workloads.build_suite(corpus, str(tmp_path / "w"), workloads.QUICK, seed=0)
    suite.main(["--workloads", str(tmp_path / "w"), "--out", str(tmp_path / "r"), "--dry-run",
                "--redline", "int4=m.rlm --draft-model d.rlm", "--hf", "org/model", "--vllm", "org/model"])
    err = capsys.readouterr().err
    bench_lines = [line for line in err.splitlines() if "redline_bench" in line]
    assert len(bench_lines) == 3  # offline b1, offline b8, online
    assert any("--online" in line for line in bench_lines)
    assert all("--kv-cache-tokens" in line and "--draft-model d.rlm" in line for line in bench_lines)
    assert sum("hf_bench" in line for line in err.splitlines()) == 1
    assert sum("vllm_bench" in line for line in err.splitlines()) == 3


def test_cli_engine_flag_mapping(monkeypatch):
    monkeypatch.setattr("redline.runtime.find_binary", lambda name: "/bin/true")
    e = CliEngine("m.rlm", max_seqs=4, cuda_graphs=False, async_scheduling=True, kv_dtype="int8",
                  track_memory=True, draft_model=None)
    assert e.flags == ["--max-seqs", "4", "--no-cuda-graphs", "--kv-dtype", "int8", "--track-memory"]
    assert os.path.basename(e.binary) == "true"
