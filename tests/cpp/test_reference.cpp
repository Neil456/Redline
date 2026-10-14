// Tests for the CPU reference model and the C++ quantizers.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "core/common.h"
#include "core/safetensors.h"
#include "cpu/random_model.h"
#include "cpu/reference_model.h"
#include "test_framework.h"

using namespace redline;

namespace {

std::string tmp_model(const std::string& tag) {
  const char* dir = std::getenv("TMPDIR");
  return std::string(dir ? dir : "/tmp") + "/redline_ref_" + tag + ".rlm";
}

std::vector<float> random_matrix(int n, int k, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.f, 0.05f);
  std::vector<float> w(static_cast<size_t>(n) * k);
  for (auto& x : w) x = dist(rng);
  return w;
}

}  // namespace

TEST(int8_quantization_error_bound) {
  const int n = 16, k = 256;
  const auto w = random_matrix(n, k, 1);
  std::vector<int8_t> q;
  std::vector<uint16_t> s;
  quantize_int8_rows(w, n, k, &q, &s);
  for (int r = 0; r < n; ++r) {
    const float scale = half_bits_to_float(s[r]);
    for (int c = 0; c < k; ++c) {
      const float deq = q[static_cast<size_t>(r) * k + c] * scale;
      EXPECT_TRUE(std::fabs(deq - w[static_cast<size_t>(r) * k + c]) <= 0.5f * scale + 1e-6f);
    }
  }
}

TEST(int4_quantization_error_bound_and_packing) {
  const int n = 8, k = 128, g = 32;
  const auto w = random_matrix(n, k, 2);
  std::vector<uint8_t> packed;
  std::vector<uint16_t> s, o;
  quantize_int4_groups(w, n, k, g, &packed, &s, &o);
  EXPECT_EQ(packed.size(), static_cast<size_t>(n * k / 2));
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < k; ++c) {
      const uint8_t byte = packed[(static_cast<size_t>(r) * k + c) / 2];
      const int qv = c % 2 == 0 ? (byte & 0xF) : (byte >> 4);
      const size_t gi = static_cast<size_t>(r) * (k / g) + c / g;
      const float scale = half_bits_to_float(s[gi]);
      const float deq = qv * scale + half_bits_to_float(o[gi]);
      EXPECT_TRUE(std::fabs(deq - w[static_cast<size_t>(r) * k + c]) <= 0.5f * scale + 2e-3f);
    }
  }
}

TEST(reference_model_causal_prefix_consistency) {
  // Logits for a prefix must equal the first rows of a longer sequence.
  const std::string path = tmp_model("causal");
  write_random_model(path, tiny_test_config(), 7);
  CpuReferenceModel ref(path);
  const int V = ref.config().vocab_size;
  const std::vector<int32_t> full = {1, 17, 300, 42, 9, 77, 5};
  const std::vector<int32_t> prefix(full.begin(), full.begin() + 4);
  const auto lf = ref.forward(full);
  const auto lp = ref.forward(prefix);
  for (size_t i = 0; i < lp.size(); ++i) EXPECT_NEAR(lp[i], lf[i], 1e-4);
  // Different positions should not produce identical rows.
  double diff = 0;
  for (int v = 0; v < V; ++v) diff += std::fabs(lf[v] - lf[static_cast<size_t>(V) + v]);
  EXPECT_TRUE(diff > 1e-3);
  std::remove(path.c_str());
}

TEST(reference_model_quantized_close_to_fp16) {
  // Same seed => same underlying fp32 weights; quantized logits stay close.
  const std::vector<int32_t> toks = {1, 5, 9, 200, 33, 18};
  std::vector<std::vector<float>> logits;
  for (QuantType q : {QuantType::kFP16, QuantType::kINT8, QuantType::kINT4}) {
    const std::string path = tmp_model(quant_type_name(q));
    write_random_model(path, tiny_test_config(q), 11);
    logits.push_back(CpuReferenceModel(path).forward(toks));
    std::remove(path.c_str());
  }
  auto rel_err = [&](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      num += (a[i] - b[i]) * (a[i] - b[i]);
      den += a[i] * a[i];
    }
    return std::sqrt(num / den);
  };
  const double e8 = rel_err(logits[0], logits[1]);
  const double e4 = rel_err(logits[0], logits[2]);
  EXPECT_TRUE(e8 < 0.02);
  EXPECT_TRUE(e4 < 0.25);
  EXPECT_TRUE(e8 < e4);
}

TEST(reference_greedy_generation_is_deterministic) {
  const std::string path = tmp_model("greedy");
  write_random_model(path, tiny_test_config(), 3);
  CpuReferenceModel ref(path);
  const auto a = ref.generate_greedy({1, 2, 3}, 5);
  const auto b = ref.generate_greedy({1, 2, 3}, 5);
  EXPECT_TRUE(a == b);
  EXPECT_EQ(a.size(), static_cast<size_t>(5));
  std::remove(path.c_str());
}
