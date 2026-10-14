#include "cpu/random_model.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "core/common.h"
#include "core/safetensors.h"

namespace redline {
namespace {

std::vector<uint16_t> to_f16(const std::vector<float>& v) {
  std::vector<uint16_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = float_to_half_bits(v[i]);
  return out;
}

class ModelBuilder {
 public:
  ModelBuilder(const ModelConfig& cfg, uint64_t seed) : cfg_(cfg), rng_(seed) {}

  std::vector<float> randn(size_t n, float stddev) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng_);
    return v;
  }

  void add_f16(const std::string& name, const std::vector<int64_t>& shape,
               const std::vector<float>& v) {
    const std::vector<uint16_t> h = to_f16(v);
    writer_.add(name, DType::kF16, shape, h.data(), h.size() * 2);
  }

  void add_linear(const std::string& prefix, int n, int k, QuantType q, bool bias,
                  float gain = 1.0f) {
    const std::vector<float> w =
        randn(static_cast<size_t>(n) * k, gain / std::sqrt(static_cast<float>(k)));
    if (q == QuantType::kFP16) {
      add_f16(prefix + ".weight", {n, k}, w);
    } else if (q == QuantType::kINT8) {
      std::vector<int8_t> qw;
      std::vector<uint16_t> s;
      quantize_int8_rows(w, n, k, &qw, &s);
      writer_.add(prefix + ".weight", DType::kI8, {n, k}, qw.data(), qw.size());
      writer_.add(prefix + ".scales", DType::kF16, {n}, s.data(), s.size() * 2);
    } else {
      std::vector<uint8_t> packed;
      std::vector<uint16_t> s, o;
      quantize_int4_groups(w, n, k, cfg_.group_size, &packed, &s, &o);
      const int64_t groups = k / cfg_.group_size;
      writer_.add(prefix + ".weight", DType::kU8, {n, k / 2}, packed.data(), packed.size());
      writer_.add(prefix + ".scales", DType::kF16, {n, groups}, s.data(), s.size() * 2);
      writer_.add(prefix + ".offsets", DType::kF16, {n, groups}, o.data(), o.size() * 2);
    }
    if (bias) add_f16(prefix + ".bias", {n}, randn(n, 0.02f));
  }

  std::vector<float> norm_weight() {
    std::uniform_real_distribution<float> dist(0.8f, 1.2f);
    std::vector<float> v(cfg_.hidden_size);
    for (auto& x : v) x = dist(rng_);
    return v;
  }

  void build(const std::string& path) {
    const int H = cfg_.hidden_size;
    add_f16("tok_embeddings.weight", {cfg_.vocab_size, H},
            randn(static_cast<size_t>(cfg_.vocab_size) * H, 1.0f));
    for (int l = 0; l < cfg_.num_layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      add_f16(p + "attn_norm.weight", {H}, norm_weight());
      add_linear(p + "wqkv", cfg_.qkv_size(), H, cfg_.weight_quant, cfg_.attention_bias);
      add_linear(p + "wo", H, cfg_.q_size(), cfg_.weight_quant, false);
      add_f16(p + "ffn_norm.weight", {H}, norm_weight());
      add_linear(p + "w_gate_up", 2 * cfg_.intermediate_size, H, cfg_.weight_quant, false);
      add_linear(p + "w_down", H, cfg_.intermediate_size, cfg_.weight_quant, false);
    }
    add_f16("norm.weight", {H}, norm_weight());
    if (!cfg_.tie_word_embeddings || cfg_.lm_head_quant != QuantType::kFP16) {
      // A larger gain gives peaked next-token distributions, so greedy
      // decisions are robust to fp16 rounding in GPU-vs-CPU comparisons.
      add_linear("lm_head", cfg_.vocab_size, H, cfg_.lm_head_quant, false, 4.0f);
    }
    writer_.set_metadata("format", "redline");
    writer_.set_metadata("redline_config", cfg_.to_json().dump());
    writer_.write(path);
  }

 private:
  ModelConfig cfg_;
  std::mt19937_64 rng_;
  SafetensorsWriter writer_;
};

}  // namespace

void quantize_int8_rows(const std::vector<float>& w, int n, int k, std::vector<int8_t>* q,
                        std::vector<uint16_t>* scales_f16) {
  q->resize(static_cast<size_t>(n) * k);
  scales_f16->resize(n);
  for (int r = 0; r < n; ++r) {
    const float* row = w.data() + static_cast<size_t>(r) * k;
    float amax = 0.f;
    for (int c = 0; c < k; ++c) amax = std::max(amax, std::fabs(row[c]));
    const uint16_t s16 = float_to_half_bits(amax > 0 ? amax / 127.f : 1.f);
    const float s = half_bits_to_float(s16);  // quantize against the stored scale
    (*scales_f16)[r] = s16;
    for (int c = 0; c < k; ++c) {
      const float v = std::nearbyint(row[c] / s);
      (*q)[static_cast<size_t>(r) * k + c] = static_cast<int8_t>(std::clamp(v, -127.f, 127.f));
    }
  }
}

void quantize_int4_groups(const std::vector<float>& w, int n, int k, int group_size,
                          std::vector<uint8_t>* packed, std::vector<uint16_t>* scales_f16,
                          std::vector<uint16_t>* offsets_f16) {
  RL_CHECK(k % group_size == 0 && group_size % 2 == 0, "bad INT4 group size");
  const int groups = k / group_size;
  packed->assign(static_cast<size_t>(n) * k / 2, 0);
  scales_f16->resize(static_cast<size_t>(n) * groups);
  offsets_f16->resize(static_cast<size_t>(n) * groups);
  for (int r = 0; r < n; ++r) {
    for (int g = 0; g < groups; ++g) {
      const float* x = w.data() + static_cast<size_t>(r) * k + static_cast<size_t>(g) * group_size;
      float lo = x[0], hi = x[0];
      for (int i = 1; i < group_size; ++i) {
        lo = std::min(lo, x[i]);
        hi = std::max(hi, x[i]);
      }
      const uint16_t s16 = float_to_half_bits(hi > lo ? (hi - lo) / 15.f : 1.f);
      const uint16_t o16 = float_to_half_bits(lo);
      const float s = half_bits_to_float(s16), o = half_bits_to_float(o16);
      (*scales_f16)[static_cast<size_t>(r) * groups + g] = s16;
      (*offsets_f16)[static_cast<size_t>(r) * groups + g] = o16;
      for (int i = 0; i < group_size; ++i) {
        const int c = g * group_size + i;
        const int qv = static_cast<int>(std::clamp(std::nearbyint((x[i] - o) / s), 0.f, 15.f));
        uint8_t& byte = (*packed)[(static_cast<size_t>(r) * k + c) / 2];
        byte |= static_cast<uint8_t>(c % 2 == 0 ? qv : (qv << 4));
      }
    }
  }
}

ModelConfig tiny_test_config(QuantType quant) {
  ModelConfig c;
  c.vocab_size = 512;
  c.hidden_size = 256;
  c.intermediate_size = 512;
  c.num_layers = 2;
  c.num_heads = 4;
  c.num_kv_heads = 2;
  c.head_dim = 64;
  c.rms_norm_eps = 1e-5;
  c.rope_theta = 10000.0;
  c.max_position_embeddings = 2048;
  c.bos_token_id = 1;
  c.eos_token_ids = {2};
  c.weight_quant = quant;
  c.lm_head_quant = QuantType::kFP16;
  c.group_size = 64;
  return c;
}

void write_random_model(const std::string& path, const ModelConfig& cfg, uint64_t seed) {
  cfg.validate();
  ModelBuilder(cfg, seed).build(path);
}

}  // namespace redline
