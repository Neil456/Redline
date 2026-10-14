#include "cpu/reference_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/common.h"
#include "core/json.h"

namespace redline {

std::vector<float> read_float_tensor(const SafetensorsFile& file, const std::string& name) {
  const TensorInfo& ti = file.info(name);
  const int64_t n = ti.numel();
  std::vector<float> out(static_cast<size_t>(n));
  const void* src = file.data(ti);
  switch (ti.dtype) {
    case DType::kF32:
      std::memcpy(out.data(), src, static_cast<size_t>(n) * 4);
      break;
    case DType::kF16: {
      const auto* h = static_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < n; ++i) out[i] = half_bits_to_float(h[i]);
      break;
    }
    case DType::kBF16: {
      const auto* h = static_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < n; ++i) {
        const uint32_t bits = static_cast<uint32_t>(h[i]) << 16;
        std::memcpy(&out[i], &bits, 4);
      }
      break;
    }
    default:
      throw Error("tensor " + name + " is not a float tensor");
  }
  return out;
}

ModelConfig read_model_config(const SafetensorsFile& file) {
  auto it = file.metadata().find("redline_config");
  RL_CHECK(it != file.metadata().end(), file.path(),
           " is not a Redline model file (missing redline_config metadata); "
           "convert it with `python -m redline.convert`");
  return ModelConfig::from_json(Json::parse(it->second));
}

std::vector<float> dequantize_linear(const SafetensorsFile& file, const std::string& prefix,
                                     QuantType qtype, int n, int k, int group_size) {
  std::vector<float> w(static_cast<size_t>(n) * k);
  const TensorInfo& wi = file.info(prefix + ".weight");
  if (qtype == QuantType::kFP16) {
    RL_CHECK(wi.shape == std::vector<int64_t>({n, k}), prefix, ".weight has shape ",
             wi.shape_str());
    return read_float_tensor(file, prefix + ".weight");
  }
  const std::vector<float> scales = read_float_tensor(file, prefix + ".scales");
  if (qtype == QuantType::kINT8) {
    RL_CHECK(wi.dtype == DType::kI8 && wi.shape == std::vector<int64_t>({n, k}),
             prefix, ".weight must be I8 [", n, ", ", k, "]");
    RL_CHECK(scales.size() == static_cast<size_t>(n), prefix, ".scales must have ", n, " entries");
    const auto* q = static_cast<const int8_t*>(file.data(wi));
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < k; ++c) {
        w[static_cast<size_t>(r) * k + c] = static_cast<float>(q[static_cast<size_t>(r) * k + c]) * scales[r];
      }
    }
    return w;
  }
  // INT4, group-wise asymmetric: w = q * scale + offset, q in [0, 15].
  RL_CHECK(wi.dtype == DType::kU8 && wi.shape == std::vector<int64_t>({n, k / 2}),
           prefix, ".weight must be U8 [", n, ", ", k / 2, "]");
  const int groups = k / group_size;
  const std::vector<float> offsets = read_float_tensor(file, prefix + ".offsets");
  RL_CHECK(scales.size() == static_cast<size_t>(n) * groups &&
               offsets.size() == static_cast<size_t>(n) * groups,
           prefix, " has wrong scale/offset sizes");
  const auto* packed = static_cast<const uint8_t*>(file.data(wi));
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < k; ++c) {
      const uint8_t byte = packed[static_cast<size_t>(r) * (k / 2) + c / 2];
      const int q = (c % 2 == 0) ? (byte & 0xF) : (byte >> 4);
      const size_t g = static_cast<size_t>(r) * groups + c / group_size;
      w[static_cast<size_t>(r) * k + c] = static_cast<float>(q) * scales[g] + offsets[g];
    }
  }
  return w;
}

CpuReferenceModel::Linear CpuReferenceModel::load_linear(const SafetensorsFile& f,
                                                         const std::string& prefix, QuantType q,
                                                         int n, int k) const {
  Linear lin;
  lin.n = n;
  lin.k = k;
  lin.w = dequantize_linear(f, prefix, q, n, k, cfg_.group_size);
  if (f.has(prefix + ".bias")) lin.bias = read_float_tensor(f, prefix + ".bias");
  return lin;
}

CpuReferenceModel::CpuReferenceModel(const std::string& path) {
  SafetensorsFile f(path);
  cfg_ = read_model_config(f);
  const int H = cfg_.hidden_size;
  embed_ = read_float_tensor(f, "tok_embeddings.weight");
  RL_CHECK(embed_.size() == static_cast<size_t>(cfg_.vocab_size) * H, "bad embedding shape");
  for (int l = 0; l < cfg_.num_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    Layer layer;
    layer.attn_norm = read_float_tensor(f, p + "attn_norm.weight");
    layer.ffn_norm = read_float_tensor(f, p + "ffn_norm.weight");
    layer.wqkv = load_linear(f, p + "wqkv", cfg_.weight_quant, cfg_.qkv_size(), H);
    layer.wo = load_linear(f, p + "wo", cfg_.weight_quant, H, cfg_.q_size());
    layer.w_gate_up =
        load_linear(f, p + "w_gate_up", cfg_.weight_quant, 2 * cfg_.intermediate_size, H);
    layer.w_down = load_linear(f, p + "w_down", cfg_.weight_quant, H, cfg_.intermediate_size);
    layers_.push_back(std::move(layer));
  }
  norm_ = read_float_tensor(f, "norm.weight");
  if (f.has("lm_head.weight")) {
    lm_head_ = load_linear(f, "lm_head", cfg_.lm_head_quant, cfg_.vocab_size, H);
  } else {
    RL_CHECK(cfg_.tie_word_embeddings, "model has no lm_head and untied embeddings");
    lm_head_.n = cfg_.vocab_size;
    lm_head_.k = H;
    lm_head_.w = embed_;
  }
  rope_max_pos_ = std::min(cfg_.max_position_embeddings, 1 << 16);
  rope_cos_sin_tables(cfg_, rope_max_pos_, &rope_cos_, &rope_sin_);
}

void CpuReferenceModel::matmul(const float* x, int m, const Linear& lin, float* y) {
  for (int i = 0; i < m; ++i) {
    const float* xr = x + static_cast<size_t>(i) * lin.k;
    for (int j = 0; j < lin.n; ++j) {
      const float* wr = lin.w.data() + static_cast<size_t>(j) * lin.k;
      double acc = 0.0;
      for (int c = 0; c < lin.k; ++c) acc += static_cast<double>(xr[c]) * wr[c];
      if (!lin.bias.empty()) acc += lin.bias[j];
      y[static_cast<size_t>(i) * lin.n + j] = static_cast<float>(acc);
    }
  }
}

void CpuReferenceModel::rmsnorm(const float* x, const std::vector<float>& w, int m,
                                float* y) const {
  const int H = cfg_.hidden_size;
  for (int i = 0; i < m; ++i) {
    const float* xr = x + static_cast<size_t>(i) * H;
    double ss = 0.0;
    for (int c = 0; c < H; ++c) ss += static_cast<double>(xr[c]) * xr[c];
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / H + cfg_.rms_norm_eps));
    for (int c = 0; c < H; ++c) y[static_cast<size_t>(i) * H + c] = xr[c] * inv * w[c];
  }
}

std::vector<float> CpuReferenceModel::forward(const std::vector<int32_t>& tokens) const {
  const int T = static_cast<int>(tokens.size());
  RL_CHECK(T > 0 && T <= rope_max_pos_, "sequence length out of range");
  const int H = cfg_.hidden_size, D = cfg_.head_dim, Hq = cfg_.num_heads, Hkv = cfg_.num_kv_heads;
  const int I = cfg_.intermediate_size, G = cfg_.gqa_group(), half = D / 2;
  const int qkv_n = cfg_.qkv_size();

  std::vector<float> h(static_cast<size_t>(T) * H), xn(h.size()), tmp(h.size());
  for (int t = 0; t < T; ++t) {
    RL_CHECK(tokens[t] >= 0 && tokens[t] < cfg_.vocab_size, "token id out of range");
    std::copy_n(embed_.data() + static_cast<size_t>(tokens[t]) * H, H, h.data() + static_cast<size_t>(t) * H);
  }
  std::vector<float> qkv(static_cast<size_t>(T) * qkv_n), attn(static_cast<size_t>(T) * Hq * D);
  std::vector<float> gu(static_cast<size_t>(T) * 2 * I), act(static_cast<size_t>(T) * I);
  std::vector<float> scores(T);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  auto rope = [&](float* v, int pos) {
    for (int i = 0; i < half; ++i) {
      const float c = rope_cos_[static_cast<size_t>(pos) * half + i];
      const float s = rope_sin_[static_cast<size_t>(pos) * half + i];
      const float x0 = v[i], x1 = v[i + half];
      v[i] = x0 * c - x1 * s;
      v[i + half] = x1 * c + x0 * s;
    }
  };

  for (const Layer& layer : layers_) {
    rmsnorm(h.data(), layer.attn_norm, T, xn.data());
    matmul(xn.data(), T, layer.wqkv, qkv.data());
    for (int t = 0; t < T; ++t) {
      float* row = qkv.data() + static_cast<size_t>(t) * qkv_n;
      for (int hh = 0; hh < Hq + Hkv; ++hh) rope(row + hh * D, t);  // q and k heads
    }
    for (int t = 0; t < T; ++t) {
      for (int hq = 0; hq < Hq; ++hq) {
        const int hk = hq / G;
        const float* q = qkv.data() + static_cast<size_t>(t) * qkv_n + hq * D;
        float mx = -INFINITY;
        for (int j = 0; j <= t; ++j) {
          const float* k = qkv.data() + static_cast<size_t>(j) * qkv_n + Hq * D + hk * D;
          float dot = 0.f;
          for (int d = 0; d < D; ++d) dot += q[d] * k[d];
          scores[j] = dot * scale;
          mx = std::max(mx, scores[j]);
        }
        double denom = 0.0;
        for (int j = 0; j <= t; ++j) {
          scores[j] = std::exp(scores[j] - mx);
          denom += scores[j];
        }
        float* out = attn.data() + (static_cast<size_t>(t) * Hq + hq) * D;
        for (int d = 0; d < D; ++d) {
          double acc = 0.0;
          for (int j = 0; j <= t; ++j) {
            const float* v = qkv.data() + static_cast<size_t>(j) * qkv_n + (Hq + Hkv) * D + hk * D;
            acc += static_cast<double>(scores[j]) * v[d];
          }
          out[d] = static_cast<float>(acc / denom);
        }
      }
    }
    matmul(attn.data(), T, layer.wo, tmp.data());
    for (size_t i = 0; i < h.size(); ++i) h[i] += tmp[i];

    rmsnorm(h.data(), layer.ffn_norm, T, xn.data());
    matmul(xn.data(), T, layer.w_gate_up, gu.data());
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < I; ++i) {
        const float g = gu[static_cast<size_t>(t) * 2 * I + i];
        const float u = gu[static_cast<size_t>(t) * 2 * I + I + i];
        act[static_cast<size_t>(t) * I + i] = g / (1.0f + std::exp(-g)) * u;
      }
    }
    matmul(act.data(), T, layer.w_down, tmp.data());
    for (size_t i = 0; i < h.size(); ++i) h[i] += tmp[i];
  }
  rmsnorm(h.data(), norm_, T, xn.data());
  std::vector<float> logits(static_cast<size_t>(T) * cfg_.vocab_size);
  matmul(xn.data(), T, lm_head_, logits.data());
  return logits;
}

std::vector<int32_t> CpuReferenceModel::generate_greedy(const std::vector<int32_t>& prompt,
                                                        int max_new_tokens) const {
  std::vector<int32_t> seq = prompt;
  std::vector<int32_t> out;
  for (int i = 0; i < max_new_tokens; ++i) {
    const std::vector<float> logits = forward(seq);
    const float* last = logits.data() + (seq.size() - 1) * static_cast<size_t>(cfg_.vocab_size);
    const int32_t next = static_cast<int32_t>(std::max_element(last, last + cfg_.vocab_size) - last);
    out.push_back(next);
    seq.push_back(next);
  }
  return out;
}

}  // namespace redline
