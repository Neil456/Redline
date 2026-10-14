// Straightforward fp32 CPU implementation of the Llama forward pass over a
// Redline model file. It is the numerical oracle for the CUDA engine tests and
// for validating the Python exporter against Hugging Face Transformers. It
// recomputes the full sequence every call and is only meant for small models.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/model_config.h"
#include "core/safetensors.h"

namespace redline {

// Dequantizes linear layer `prefix` (e.g. "layers.0.wqkv") to fp32 [n, k].
std::vector<float> dequantize_linear(const SafetensorsFile& file, const std::string& prefix,
                                     QuantType qtype, int n, int k, int group_size);

class CpuReferenceModel {
 public:
  explicit CpuReferenceModel(const std::string& path);

  const ModelConfig& config() const { return cfg_; }

  // Returns logits [tokens.size(), vocab_size], row-major.
  std::vector<float> forward(const std::vector<int32_t>& tokens) const;

  // Greedy generation by full recomputation.
  std::vector<int32_t> generate_greedy(const std::vector<int32_t>& prompt, int max_new_tokens) const;

 private:
  struct Linear {
    int n = 0, k = 0;
    std::vector<float> w;     // [n, k]
    std::vector<float> bias;  // [n] or empty
  };
  struct Layer {
    std::vector<float> attn_norm, ffn_norm;
    Linear wqkv, wo, w_gate_up, w_down;
  };

  Linear load_linear(const SafetensorsFile& f, const std::string& prefix, QuantType q, int n,
                     int k) const;
  static void matmul(const float* x, int m, const Linear& lin, float* y);
  void rmsnorm(const float* x, const std::vector<float>& w, int m, float* y) const;

  ModelConfig cfg_;
  std::vector<float> embed_;  // [vocab, hidden]
  std::vector<Layer> layers_;
  std::vector<float> norm_;
  Linear lm_head_;
  std::vector<float> rope_cos_, rope_sin_;
  int rope_max_pos_ = 0;
};

// Reads tensor `name` as fp32 regardless of its stored float type.
std::vector<float> read_float_tensor(const SafetensorsFile& file, const std::string& name);

// Loads the ModelConfig stored in a Redline model file's metadata.
ModelConfig read_model_config(const SafetensorsFile& file);

}  // namespace redline
