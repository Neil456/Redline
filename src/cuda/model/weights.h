// Device-resident model weights loaded from a Redline model file.
#pragma once

#include <cuda_fp16.h>

#include <memory>
#include <string>
#include <vector>

#include "core/model_config.h"
#include "cuda/device_memory.h"
#include "cuda/gemm/linear.h"

namespace redline {

struct LayerWeights {
  const half* attn_norm = nullptr;
  LinearWeight wqkv;
  LinearWeight wo;
  const half* ffn_norm = nullptr;
  LinearWeight w_gate_up;
  LinearWeight w_down;
};

class ModelWeights {
 public:
  // Uploads all tensors into one device allocation through a pair of pinned
  // staging buffers (double-buffered async copies). When `trtllm` is given,
  // quantized linears are converted to TensorRT-LLM's layout instead.
  static std::unique_ptr<ModelWeights> load(const std::string& path, MemoryLedger* ledger,
                                            TrtllmWeightOnlyGemm* trtllm = nullptr);

  const ModelConfig& config() const { return config_; }
  size_t bytes() const { return storage_.bytes(); }

  const half* tok_embeddings = nullptr;
  std::vector<LayerWeights> layers;
  const half* norm = nullptr;
  LinearWeight lm_head;

 private:
  ModelConfig config_;
  DeviceBuffer storage_;
};

}  // namespace redline
