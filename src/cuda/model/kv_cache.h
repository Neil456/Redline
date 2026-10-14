// GPU-resident paged KV cache: one allocation holding every layer's K and V
// blocks (and per-token scales for INT8). Block ids are shared with the host
// BlockAllocator; a target and a draft model may use the same block ids with
// separate pools.
#pragma once

#include <string>
#include <vector>

#include "cuda/device_memory.h"
#include "cuda/kernels/kernels.h"

namespace redline {

struct KVCacheConfig {
  int num_layers = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  int block_size = 16;
  int num_blocks = 0;
  KVDType dtype = KVDType::kFP16;

  // Bytes of one block across all layers (K + V [+ scales]).
  size_t bytes_per_block() const;
};

class KVCache {
 public:
  KVCache(const KVCacheConfig& cfg, MemoryLedger* ledger, const char* category = "kv_cache");

  const KVCacheConfig& config() const { return cfg_; }
  KVLayerView layer(int l) const { return views_[l]; }
  size_t bytes() const { return storage_.bytes(); }

 private:
  KVCacheConfig cfg_;
  DeviceBuffer storage_;
  std::vector<KVLayerView> views_;
};

const char* kv_dtype_name(KVDType t);
KVDType kv_dtype_from_name(const std::string& s);

}  // namespace redline
