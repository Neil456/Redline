// Per-op GPU time of eager forward passes, from CUDA events recorded between consecutive ops.
//
// The host enqueues a whole step ahead of the GPU, so the gap between two events is the GPU
// time of the op in between (plus its launch gap). Graph replays cannot be split this way, so
// the engine runs eagerly while profiling is on. Used by `redline_bench --profile-ops` to decide
// what to fuse.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "core/json.h"
#include "cuda/cuda_utils.h"

namespace redline {

class OpProfiler {
 public:
  OpProfiler() = default;
  OpProfiler(const OpProfiler&) = delete;
  OpProfiler& operator=(const OpProfiler&) = delete;
  ~OpProfiler() {
    for (cudaEvent_t e : pool_) cudaEventDestroy(e);
  }

  // Starts a forward pass; ops are attributed to "<scope>/<op>".
  void begin(const std::string& scope, cudaStream_t s) {
    scope_ = scope;
    used_ = 0;
    names_.clear();
    record("", s);
  }
  // Marks the end of `op` (everything enqueued since the previous mark).
  void mark(const char* op, cudaStream_t s) { record(scope_ + "/" + op, s); }
  // Waits for the pass and accumulates the per-op times.
  void end() {
    if (used_ < 2) return;
    RL_CUDA_CHECK(cudaEventSynchronize(pool_[used_ - 1]));
    for (size_t i = 1; i < used_; ++i) {
      float ms = 0;
      RL_CUDA_CHECK(cudaEventElapsedTime(&ms, pool_[i - 1], pool_[i]));
      Stat& st = stats_[names_[i]];
      if (st.calls == 0) order_.push_back(names_[i]);
      st.ms += ms;
      st.calls += 1;
    }
    passes_[scope_] += 1;
    used_ = 0;
  }
  void reset() {
    stats_.clear();
    order_.clear();
    passes_.clear();
  }

  // [{op, ms_per_pass, calls_per_pass, pct}] in first-seen order; pct is within the scope.
  Json report() const {
    std::map<std::string, double> scope_ms;
    for (const auto& kv : stats_) scope_ms[kv.first.substr(0, kv.first.find('/'))] += kv.second.ms;
    Json rows = Json::array();
    for (const auto& name : order_) {
      const Stat& st = stats_.at(name);
      const std::string scope = name.substr(0, name.find('/'));
      const double passes = std::max(1, passes_.count(scope) ? passes_.at(scope) : 1);
      Json r = Json::object();
      r.set("op", name);
      r.set("ms_per_pass", st.ms / passes);
      r.set("calls_per_pass", st.calls / passes);
      r.set("pct", scope_ms[scope] > 0 ? 100.0 * st.ms / scope_ms[scope] : 0.0);
      rows.push_back(r);
    }
    return rows;
  }

 private:
  struct Stat {
    double ms = 0;
    int64_t calls = 0;
  };

  void record(std::string name, cudaStream_t s) {
    if (used_ == pool_.size()) {
      cudaEvent_t e;
      RL_CUDA_CHECK(cudaEventCreate(&e));
      pool_.push_back(e);
    }
    RL_CUDA_CHECK(cudaEventRecord(pool_[used_], s));
    names_.push_back(std::move(name));
    ++used_;
  }

  std::string scope_;
  std::vector<cudaEvent_t> pool_;
  std::vector<std::string> names_;
  size_t used_ = 0;
  std::map<std::string, Stat> stats_;
  std::vector<std::string> order_;
  std::map<std::string, int> passes_;
};

}  // namespace redline
