// Benchmark workload description (JSON). A workload is a list of requests
// with explicit prompt token ids (produced by python/redline/workloads.py
// from real datasets) or synthetic prompt lengths, plus arrival times for
// online-serving style runs.
#pragma once

#include <string>
#include <vector>

#include "core/json.h"
#include "core/sequence.h"

namespace redline {

struct Workload {
  std::string name;
  std::string description;
  std::vector<Request> requests;  // arrival_time relative to benchmark start
  Json raw_settings;              // echoed into the report

  int64_t total_prompt_tokens() const;
  int64_t total_max_new_tokens() const;
};

// Parses a workload JSON document. Requests may specify "prompt_ids" or
// "prompt_len" (random ids in [vocab_low, vocab_high) from a seeded RNG).
Workload parse_workload(const Json& doc, int vocab_size);
Workload load_workload(const std::string& path, int vocab_size);

// Fixed-shape synthetic workload: `num_requests` prompts of `prompt_len`
// random tokens, each generating exactly `output_len` tokens.
Workload synthetic_workload(int num_requests, int prompt_len, int output_len, int vocab_size,
                            uint64_t seed);

}  // namespace redline
