#include "cuda/engine/device_batch.h"

namespace redline {

DeviceBatch DeviceBatch::bind(void* device_meta, const BatchShape& shape, int max_blocks_per_seq) {
  const MetaLayout m = MetaLayout::make(shape, max_blocks_per_seq);
  auto* w = static_cast<int32_t*>(device_meta);
  DeviceBatch b;
  b.shape = shape;
  b.max_blocks_per_seq = max_blocks_per_seq;
  b.input_ids = w + m.input_ids;
  b.input_src = w + m.input_src;
  b.positions = w + m.positions;
  b.slot_mapping = w + m.slot_mapping;
  b.seq_q_start = w + m.seq_q_start;
  b.seq_q_len = w + m.seq_q_len;
  b.seq_ctx_len = w + m.seq_ctx_len;
  b.block_tables = w + m.block_tables;
  b.decode_token = w + m.decode_token;
  b.decode_seq = w + m.decode_seq;
  b.prefill_seq = w + m.prefill_seq;
  b.logit_token = w + m.logit_token;
  b.param_row = w + m.param_row;
  b.sampling.temperature = reinterpret_cast<const float*>(w + m.temperature);
  b.sampling.top_k = w + m.top_k;
  b.sampling.top_p = reinterpret_cast<const float*>(w + m.top_p);
  b.sampling.seed_lo = reinterpret_cast<const uint32_t*>(w + m.seed_lo);
  b.sampling.seed_hi = reinterpret_cast<const uint32_t*>(w + m.seed_hi);
  b.sampling.param_row = b.param_row;
  b.sampling.token_of_row = b.logit_token;
  b.sampling.positions = b.positions;
  return b;
}

}  // namespace redline
