// Latency statistics helpers for the benchmark harness.
#pragma once

#include <cstddef>
#include <vector>

#include "core/json.h"

namespace redline {

struct Summary {
  size_t count = 0;
  double mean = 0, min = 0, max = 0, p50 = 0, p90 = 0, p95 = 0, p99 = 0, stddev = 0;
  Json to_json() const;
};

// Percentiles use linear interpolation between closest ranks (numpy default).
Summary summarize(std::vector<double> values);
double percentile(std::vector<double> sorted_values, double q);

}  // namespace redline
