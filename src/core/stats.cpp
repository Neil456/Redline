#include "core/stats.h"

#include <algorithm>
#include <cmath>

namespace redline {

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q / 100.0 * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return v[lo] + (v[hi] - v[lo]) * frac;
}

Summary summarize(std::vector<double> v) {
  Summary s;
  s.count = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / static_cast<double>(v.size());
  double var = 0;
  for (double x : v) var += (x - s.mean) * (x - s.mean);
  s.stddev = std::sqrt(var / static_cast<double>(v.size()));
  s.min = v.front();
  s.max = v.back();
  s.p50 = percentile(v, 50);
  s.p90 = percentile(v, 90);
  s.p95 = percentile(v, 95);
  s.p99 = percentile(v, 99);
  return s;
}

Json Summary::to_json() const {
  Json j = Json::object();
  j.set("count", static_cast<int64_t>(count));
  j.set("mean", mean);
  j.set("p50", p50);
  j.set("p90", p90);
  j.set("p95", p95);
  j.set("p99", p99);
  j.set("min", min);
  j.set("max", max);
  j.set("std", stddev);
  return j;
}

}  // namespace redline
