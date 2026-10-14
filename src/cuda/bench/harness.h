// Benchmark harness: replays a workload through the engine and reports
// latency (TTFT, inter-token latency, TPOT, end-to-end), throughput
// (prefill / decode / output), GPU step times from CUDA events, host
// overhead, peak device memory and speculative acceptance rate.
#pragma once

#include <string>

#include "core/json.h"
#include "core/workload.h"
#include "cuda/engine/engine.h"

namespace redline {

struct BenchOptions {
  // Honor per-request arrival times (online serving). Otherwise every request
  // arrives at t=0 (offline batch).
  bool online = false;
  // Warm-up requests (taken from the workload, truncated outputs) run before
  // measurement so lazy initialization does not pollute the numbers.
  int warmup_requests = 2;
  int warmup_tokens = 16;
  bool keep_per_request = false;  // include per-request metrics in the report
  // When set, writes a step-by-step trace (scheduler state, CUDA-event step
  // times, request lifetimes) for the demo's trace viewer (demo/index.html).
  std::string trace_path;
};

Json run_benchmark(Engine& engine, const Workload& workload, const BenchOptions& opts);

// One-line human-readable summary of a report.
std::string format_report_summary(const Json& report);

}  // namespace redline
