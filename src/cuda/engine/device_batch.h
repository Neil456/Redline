// Device view of a batch uploaded by the engine (see core/batch_builder.h).
#pragma once

#include <cstdint>

#include "core/batch_builder.h"
#include "cuda/kernels/kernels.h"

namespace redline {

struct DeviceBatch {
  BatchShape shape;
  int max_blocks_per_seq = 0;
  int32_t* input_ids = nullptr;  // resolved in place from input_src on the GPU
  const int32_t* input_src = nullptr;
  const int32_t* positions = nullptr;
  const int32_t* slot_mapping = nullptr;
  const int32_t* seq_q_start = nullptr;
  const int32_t* seq_q_len = nullptr;
  const int32_t* seq_ctx_len = nullptr;
  const int32_t* block_tables = nullptr;
  const int32_t* decode_token = nullptr;
  const int32_t* decode_seq = nullptr;
  const int32_t* prefill_seq = nullptr;
  const int32_t* logit_token = nullptr;
  const int32_t* param_row = nullptr;
  SamplingParamsDevice sampling;  // one row per logits row

  static DeviceBatch bind(void* device_meta, const BatchShape& shape, int max_blocks_per_seq);
};

}  // namespace redline
