#include "core/model_config.h"

#include <cmath>

#include "core/common.h"

namespace redline {

const char* quant_type_name(QuantType q) {
  switch (q) {
    case QuantType::kFP16: return "fp16";
    case QuantType::kINT8: return "int8";
    case QuantType::kINT4: return "int4";
  }
  return "?";
}

QuantType quant_type_from_name(const std::string& name) {
  if (name == "fp16" || name == "none") return QuantType::kFP16;
  if (name == "int8") return QuantType::kINT8;
  if (name == "int4") return QuantType::kINT4;
  throw Error("unknown quantization type: " + name);
}

ModelConfig ModelConfig::from_json(const Json& j) {
  ModelConfig c;
  c.architecture = j.get_string("architecture", "llama");
  c.vocab_size = static_cast<int>(j["vocab_size"].as_int());
  c.hidden_size = static_cast<int>(j["hidden_size"].as_int());
  c.intermediate_size = static_cast<int>(j["intermediate_size"].as_int());
  c.num_layers = static_cast<int>(j["num_layers"].as_int());
  c.num_heads = static_cast<int>(j["num_heads"].as_int());
  c.num_kv_heads = static_cast<int>(j.get_int("num_kv_heads", c.num_heads));
  c.head_dim = static_cast<int>(j.get_int("head_dim", c.hidden_size / c.num_heads));
  c.rms_norm_eps = j.get_double("rms_norm_eps", 1e-5);
  c.rope_theta = j.get_double("rope_theta", 10000.0);
  if (const Json* rs = j.find("rope_scaling"); rs != nullptr && rs->is_object()) {
    c.rope_scaling.type = rs->get_string("type", "none");
    c.rope_scaling.factor = rs->get_double("factor", 1.0);
    c.rope_scaling.low_freq_factor = rs->get_double("low_freq_factor", 1.0);
    c.rope_scaling.high_freq_factor = rs->get_double("high_freq_factor", 4.0);
    c.rope_scaling.original_max_position_embeddings =
        static_cast<int>(rs->get_int("original_max_position_embeddings", 8192));
  }
  c.max_position_embeddings = static_cast<int>(j.get_int("max_position_embeddings", 4096));
  c.attention_bias = j.get_bool("attention_bias", false);
  c.tie_word_embeddings = j.get_bool("tie_word_embeddings", false);
  c.bos_token_id = static_cast<int>(j.get_int("bos_token_id", -1));
  if (const Json* eos = j.find("eos_token_ids"); eos != nullptr && eos->is_array()) {
    for (const auto& e : eos->as_array()) c.eos_token_ids.push_back(static_cast<int>(e.as_int()));
  }
  if (const Json* q = j.find("quant"); q != nullptr && q->is_object()) {
    c.weight_quant = quant_type_from_name(q->get_string("weight_dtype", "fp16"));
    c.lm_head_quant = quant_type_from_name(q->get_string("lm_head_dtype", "fp16"));
    c.group_size = static_cast<int>(q->get_int("group_size", 128));
  }
  c.validate();
  return c;
}

Json ModelConfig::to_json() const {
  Json j = Json::object();
  j.set("architecture", architecture);
  j.set("vocab_size", vocab_size);
  j.set("hidden_size", hidden_size);
  j.set("intermediate_size", intermediate_size);
  j.set("num_layers", num_layers);
  j.set("num_heads", num_heads);
  j.set("num_kv_heads", num_kv_heads);
  j.set("head_dim", head_dim);
  j.set("rms_norm_eps", rms_norm_eps);
  j.set("rope_theta", rope_theta);
  if (rope_scaling.type != "none") {
    Json rs = Json::object();
    rs.set("type", rope_scaling.type);
    rs.set("factor", rope_scaling.factor);
    rs.set("low_freq_factor", rope_scaling.low_freq_factor);
    rs.set("high_freq_factor", rope_scaling.high_freq_factor);
    rs.set("original_max_position_embeddings", rope_scaling.original_max_position_embeddings);
    j.set("rope_scaling", rs);
  } else {
    j.set("rope_scaling", Json());
  }
  j.set("max_position_embeddings", max_position_embeddings);
  j.set("attention_bias", attention_bias);
  j.set("tie_word_embeddings", tie_word_embeddings);
  j.set("bos_token_id", bos_token_id);
  j.set("eos_token_ids", Json::array_of(eos_token_ids));
  Json q = Json::object();
  q.set("weight_dtype", quant_type_name(weight_quant));
  q.set("lm_head_dtype", quant_type_name(lm_head_quant));
  q.set("group_size", group_size);
  j.set("quant", q);
  return j;
}

void ModelConfig::validate() const {
  RL_CHECK(architecture == "llama", "unsupported architecture: ", architecture);
  RL_CHECK(vocab_size > 0 && hidden_size > 0 && intermediate_size > 0 && num_layers > 0,
           "invalid model dimensions");
  RL_CHECK(num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0,
           "num_heads must be a multiple of num_kv_heads");
  RL_CHECK(head_dim % 2 == 0, "head_dim must be even");
  RL_CHECK(hidden_size % 32 == 0 && intermediate_size % 32 == 0,
           "hidden/intermediate sizes must be multiples of 32");
  if (weight_quant == QuantType::kINT4 || lm_head_quant == QuantType::kINT4) {
    RL_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
             "INT4 group_size must be 32, 64 or 128");
    RL_CHECK(hidden_size % group_size == 0 && intermediate_size % group_size == 0 &&
                 q_size() % group_size == 0,
             "dimensions must be divisible by group_size");
  }
  RL_CHECK(rope_scaling.type == "none" || rope_scaling.type == "linear" ||
               rope_scaling.type == "llama3" || rope_scaling.type == "default",
           "unsupported rope_scaling type: ", rope_scaling.type);
}

std::string ModelConfig::summary() const {
  return str_cat("llama L=", num_layers, " H=", hidden_size, " I=", intermediate_size,
                 " heads=", num_heads, "/", num_kv_heads, " head_dim=", head_dim,
                 " vocab=", vocab_size, " weights=", quant_type_name(weight_quant),
                 " lm_head=", quant_type_name(lm_head_quant),
                 weight_quant == QuantType::kINT4 ? str_cat(" g=", group_size) : "");
}

std::vector<double> rope_inv_freq(const ModelConfig& cfg) {
  const int half = cfg.head_dim / 2;
  std::vector<double> inv(half);
  for (int i = 0; i < half; ++i) {
    inv[i] = 1.0 / std::pow(cfg.rope_theta, (2.0 * i) / cfg.head_dim);
  }
  const RopeScaling& rs = cfg.rope_scaling;
  if (rs.type == "linear") {
    for (auto& f : inv) f /= rs.factor;
  } else if (rs.type == "llama3") {
    // Mirrors transformers' _compute_llama3_parameters.
    const double old_ctx = rs.original_max_position_embeddings;
    const double low_freq_wavelen = old_ctx / rs.low_freq_factor;
    const double high_freq_wavelen = old_ctx / rs.high_freq_factor;
    for (auto& f : inv) {
      const double wavelen = 2.0 * M_PI / f;
      if (wavelen < high_freq_wavelen) continue;
      if (wavelen > low_freq_wavelen) {
        f = f / rs.factor;
      } else {
        const double smooth =
            (old_ctx / wavelen - rs.low_freq_factor) / (rs.high_freq_factor - rs.low_freq_factor);
        f = (1.0 - smooth) * f / rs.factor + smooth * f;
      }
    }
  }
  return inv;
}

void rope_cos_sin_tables(const ModelConfig& cfg, int max_pos, std::vector<float>* cos_table,
                         std::vector<float>* sin_table) {
  const std::vector<double> inv = rope_inv_freq(cfg);
  const int half = cfg.head_dim / 2;
  cos_table->resize(static_cast<size_t>(max_pos) * half);
  sin_table->resize(static_cast<size_t>(max_pos) * half);
  for (int p = 0; p < max_pos; ++p) {
    for (int i = 0; i < half; ++i) {
      // HF computes the angle in fp32; do the same so long contexts agree.
      const float angle = static_cast<float>(p) * static_cast<float>(inv[i]);
      (*cos_table)[static_cast<size_t>(p) * half + i] = std::cos(angle);
      (*sin_table)[static_cast<size_t>(p) * half + i] = std::sin(angle);
    }
  }
}

}  // namespace redline
