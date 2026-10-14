#include "cuda/model/kv_cache.h"

#include <string>

#include "cuda/cuda_utils.h"

namespace redline {

const char* kv_dtype_name(KVDType t) { return t == KVDType::kFP16 ? "fp16" : "int8"; }

KVDType kv_dtype_from_name(const std::string& s) {
  if (s == "fp16") return KVDType::kFP16;
  if (s == "int8") return KVDType::kINT8;
  throw Error("unknown KV cache dtype: " + s);
}

size_t KVCacheConfig::bytes_per_block() const {
  const size_t cells = static_cast<size_t>(num_kv_heads) * block_size;
  const size_t elem = dtype == KVDType::kFP16 ? 2 : 1;
  size_t per_layer = 2 * cells * head_dim * elem;
  if (dtype == KVDType::kINT8) per_layer += 2 * cells * sizeof(float);
  return per_layer * num_layers;
}

KVCache::KVCache(const KVCacheConfig& cfg, MemoryLedger* ledger, const char* category)
    : cfg_(cfg) {
  RL_CHECK(cfg.num_blocks > 1 && cfg.block_size > 0 && cfg.num_layers > 0, "bad KV cache config");
  storage_ = ledger->alloc(category, cfg.bytes_per_block() * cfg.num_blocks);
  // Zero once so never-written cells (e.g. the scratch block) hold finite values.
  RL_CUDA_CHECK(cudaMemset(storage_.get(), 0, storage_.bytes()));
  const size_t cells = static_cast<size_t>(cfg.num_blocks) * cfg.num_kv_heads * cfg.block_size;
  const size_t elem = cfg.dtype == KVDType::kFP16 ? 2 : 1;
  auto* p = static_cast<uint8_t*>(storage_.get());
  for (int l = 0; l < cfg.num_layers; ++l) {
    KVLayerView v;
    v.block_size = cfg.block_size;
    v.num_kv_heads = cfg.num_kv_heads;
    v.head_dim = cfg.head_dim;
    v.dtype = cfg.dtype;
    v.k = p;
    p += cells * cfg.head_dim * elem;
    v.v = p;
    p += cells * cfg.head_dim * elem;
    if (cfg.dtype == KVDType::kINT8) {
      v.k_scale = reinterpret_cast<float*>(p);
      p += cells * sizeof(float);
      v.v_scale = reinterpret_cast<float*>(p);
      p += cells * sizeof(float);
    }
    views_.push_back(v);
  }
}

}  // namespace redline
