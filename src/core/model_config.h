// Model hyper-parameters stored in the Redline model file metadata.
#pragma once

#include <string>
#include <vector>

#include "core/json.h"

namespace redline {

// Weight storage format of a linear layer.
//   kFP16: half [N, K]
//   kINT8: int8 [N, K] + per-output-channel half scales [N]   (w = q * s)
//   kINT4: uint8 [N, K/2] (two nibbles, low = even k) + half scales/offsets
//          [N, K/group]                                      (w = q * s + o)
enum class QuantType { kFP16 = 0, kINT8 = 1, kINT4 = 2 };

const char* quant_type_name(QuantType q);
QuantType quant_type_from_name(const std::string& name);

struct RopeScaling {
  std::string type = "none";  // none | linear | llama3
  double factor = 1.0;
  double low_freq_factor = 1.0;
  double high_freq_factor = 4.0;
  int original_max_position_embeddings = 8192;
};

struct ModelConfig {
  std::string architecture = "llama";
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  double rms_norm_eps = 1e-5;
  double rope_theta = 10000.0;
  RopeScaling rope_scaling;
  int max_position_embeddings = 4096;
  bool attention_bias = false;
  bool tie_word_embeddings = false;
  int bos_token_id = -1;
  std::vector<int> eos_token_ids;

  QuantType weight_quant = QuantType::kFP16;
  QuantType lm_head_quant = QuantType::kFP16;
  int group_size = 128;

  int q_size() const { return num_heads * head_dim; }
  int kv_size() const { return num_kv_heads * head_dim; }
  int qkv_size() const { return q_size() + 2 * kv_size(); }
  int gqa_group() const { return num_heads / num_kv_heads; }

  static ModelConfig from_json(const Json& j);
  Json to_json() const;
  void validate() const;
  std::string summary() const;
};

// Rotary inverse frequencies [head_dim / 2], including rope scaling.
std::vector<double> rope_inv_freq(const ModelConfig& cfg);

// cos/sin tables of shape [max_pos, head_dim / 2] (row-major), matching
// HF's rotate_half convention: pair (i, i + head_dim/2) rotates by angle
// pos * inv_freq[i].
void rope_cos_sin_tables(const ModelConfig& cfg, int max_pos, std::vector<float>* cos_table,
                         std::vector<float>* sin_table);

}  // namespace redline
