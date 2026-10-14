// Llama-family transformer forward pass on the GPU.
#pragma once

#include <cuda_fp16.h>

#include "cuda/device_memory.h"
#include "cuda/engine/device_batch.h"
#include "cuda/gemm/linear.h"
#include "cuda/model/kv_cache.h"
#include "cuda/model/weights.h"

namespace redline {

struct ModelRuntimeConfig {
  int max_tokens = 2048;       // tokens per forward step
  int max_logit_rows = 256;    // logits rows per forward step
  int max_seq_len = 4096;      // RoPE table length and split-K partition bound
  int max_decode_items = 256;  // query tokens served by decode attention per step
  bool nvtx_per_layer = false;
};

class LlamaModel {
 public:
  LlamaModel(const ModelWeights* weights, LinearExecutor* linear, const ModelRuntimeConfig& rc,
             MemoryLedger* ledger, const char* name);

  // The KV cache is sized after activations are allocated, so it is bound
  // separately (before the first forward).
  void bind_kv_cache(const KVCache* kv);

  // Embedding -> decoder layers -> final norm -> lm_head for the batch's
  // logits rows. `logits` is [num_logit_rows, vocab] fp32; when null the
  // lm_head is skipped (a draft-model prefill only fills its KV cache).
  // With `static_partitions` the decode attention grid covers max_seq_len
  // (required for CUDA-graph capture).
  void forward(const DeviceBatch& batch, float* logits, bool static_partitions,
               cudaStream_t stream);

  const ModelConfig& config() const { return weights_->config(); }
  const ModelWeights& weights() const { return *weights_; }
  const ModelRuntimeConfig& runtime() const { return rc_; }
  int max_partitions() const;

 private:
  const ModelWeights* weights_;
  const KVCache* kv_ = nullptr;
  LinearExecutor* linear_;
  ModelRuntimeConfig rc_;
  const char* name_;
  float attn_scale_;

  DeviceBuffer rope_cos_, rope_sin_;
  DeviceBuffer residual_, normed_;
  DeviceBuffer buf_a_;  // qkv, then gate_up
  DeviceBuffer buf_b_;  // attention output, then SwiGLU activations
  DeviceBuffer buf_c_;  // o_proj / down_proj outputs
  DeviceBuffer logit_hidden_;
  DeviceBuffer attn_ws_out_, attn_ws_max_, attn_ws_sum_;
};

}  // namespace redline
