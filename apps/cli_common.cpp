#include "cli_common.h"

#include <sstream>

namespace redline::cli {

Args::Args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) throw Error("unexpected argument: " + a);
    a = a.substr(2);
    const auto eq = a.find('=');
    if (eq != std::string::npos) {
      kv_.emplace(a.substr(0, eq), a.substr(eq + 1));
    } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      kv_.emplace(a, argv[++i]);
    } else {
      kv_.emplace(a, "");
    }
  }
}

bool Args::has(const std::string& key) const {
  used_[key] = true;
  return kv_.count(key) != 0;
}

std::string Args::get(const std::string& key, const std::string& def) const {
  used_[key] = true;
  auto it = kv_.find(key);
  return it == kv_.end() ? def : it->second;
}

int Args::get_int(const std::string& key, int def) const {
  const std::string v = get(key);
  return v.empty() ? def : std::stoi(v);
}

double Args::get_double(const std::string& key, double def) const {
  const std::string v = get(key);
  return v.empty() ? def : std::stod(v);
}

std::vector<std::string> Args::get_all(const std::string& key) const {
  used_[key] = true;
  std::vector<std::string> out;
  auto range = kv_.equal_range(key);
  for (auto it = range.first; it != range.second; ++it) out.push_back(it->second);
  return out;
}

void Args::check_unknown() const {
  for (const auto& kv : kv_) {
    if (!used_.count(kv.first)) throw Error("unknown option --" + kv.first);
  }
}

std::vector<int32_t> parse_id_list(const std::string& s) {
  std::vector<int32_t> ids;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) ids.push_back(static_cast<int32_t>(std::stol(tok)));
  }
  return ids;
}

EngineConfig engine_config_from_args(const Args& a) {
  EngineConfig c;
  c.model_path = a.get("model");
  if (c.model_path.empty()) throw Error("--model is required");
  c.draft_model_path = a.get("draft-model");
  c.num_speculative_tokens = a.get_int("spec-tokens", c.num_speculative_tokens);
  c.max_num_seqs = a.get_int("max-seqs", c.max_num_seqs);
  c.max_num_batched_tokens = a.get_int("max-batched-tokens", c.max_num_batched_tokens);
  c.max_seq_len = a.get_int("max-seq-len", c.max_seq_len);
  c.max_logit_rows = a.get_int("max-logit-rows", c.max_logit_rows);
  c.block_size = a.get_int("block-size", c.block_size);
  c.kv_cache_tokens = a.get_int("kv-cache-tokens", 0);
  c.gpu_memory_utilization = a.get_double("gpu-memory-utilization", c.gpu_memory_utilization);
  c.kv_dtype = kv_dtype_from_name(a.get("kv-dtype", "fp16"));
  c.cuda_graphs = !a.flag("no-cuda-graphs");
  c.async_scheduling = !a.flag("no-async");
  c.mix_prefill_decode = !a.flag("no-mixed-batches");
  c.linear.gemv_max_m = a.get_int("gemv-max-m", c.linear.gemv_max_m);
  c.linear.int8_prefill = prefill_gemm_from_name(a.get("int8-gemm", "auto"));
  c.linear.int4_prefill = prefill_gemm_from_name(a.get("int4-gemm", "auto"));
  c.nvtx_per_layer = a.flag("nvtx-layers");
  c.track_memory = a.flag("track-memory");
  c.device = a.get_int("device", 0);
  return c;
}

const char* engine_options_help() {
  return R"(engine options:
  --model PATH                 Redline model file (.rlm)            [required]
  --draft-model PATH           draft model for speculative decoding
  --spec-tokens K              draft tokens per step (default 4)
  --max-seqs N                 max concurrent sequences (default 32)
  --max-batched-tokens N       token budget per step / prefill chunk (default 2048)
  --max-seq-len N              max prompt + output length (default 4096)
  --kv-cache-tokens N          KV capacity in tokens (default: from memory budget)
  --gpu-memory-utilization F   device memory fraction to use (default 0.90)
  --kv-dtype fp16|int8         KV cache precision (default fp16)
  --int8-gemm auto|dequant|w8a8|cutlass|trtllm   batched INT8 GEMM backend
  --int4-gemm auto|dequant|trtllm                batched INT4 GEMM backend
  --gemv-max-m N               rows handled by weight-only GEMV kernels (default 8)
  --no-cuda-graphs             disable CUDA graph replay for decode
  --no-async                   disable asynchronous (pipelined) scheduling
  --no-mixed-batches           do not mix prefill chunks with decode tokens
  --nvtx-layers                emit one NVTX range per transformer layer
  --track-memory               sample device memory every step (peak VRAM)
  --device N                   CUDA device (default 0)
)";
}

}  // namespace redline::cli
