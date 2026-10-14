// Per-step batch metadata. The host builds every index array a forward step
// needs (token ids, positions, KV slots, block tables, attention work lists,
// logits rows, sampling parameters) directly into a pinned staging buffer,
// which is uploaded with a single async memcpy into a fixed device buffer.
// The layout depends only on the batch shape, so a CUDA graph captured for a
// decode batch size is replayed by refreshing the buffer contents.
#pragma once

#include <cstdint>
#include <vector>

#include "core/sequence.h"

namespace redline {

struct BatchShape {
  int num_tokens = 0;        // T
  int num_seqs = 0;          // S
  int num_logit_rows = 0;    // R
  int num_decode_items = 0;  // ND: tokens served by the decode attention kernel
  int num_prefill_seqs = 0;  // NP: sequences served by the prefill attention kernel
  int max_prefill_q_tiles = 0;
  int max_context = 0;       // longest context in the step

  bool operator==(const BatchShape& o) const;
};

// Word (4-byte) offsets of each array inside the metadata buffer.
struct MetaLayout {
  size_t input_ids = 0, input_src = 0, positions = 0, slot_mapping = 0;  // [T]
  size_t seq_q_start = 0, seq_q_len = 0, seq_ctx_len = 0;                // [S]
  size_t block_tables = 0;                                               // [S, max_blocks]
  size_t decode_token = 0, decode_seq = 0;                               // [ND]
  size_t prefill_seq = 0;                                                // [NP]
  size_t logit_token = 0, param_row = 0;                                 // [R]
  size_t temperature = 0, top_k = 0, top_p = 0, seed_lo = 0, seed_hi = 0;  // [S]
  size_t total_words = 0;

  static MetaLayout make(const BatchShape& shape, int max_blocks_per_seq);
};

enum class LogitsMode { kNone, kLast, kAll };

// Query-tile height of the prefill attention kernel (kernels::kPrefillTileQ).
constexpr int kBatchPrefillTileQ = 32;

class BatchBuilder {
 public:
  // Sequences adding at most `decode_query_threshold` tokens use the decode
  // attention kernel (one work item per token); longer ones use prefill tiles.
  BatchBuilder(int max_tokens, int max_seqs, int max_logit_rows, int max_blocks_per_seq,
               int block_size, int decode_query_threshold);

  void reset();

  // Adds positions [start_pos, start_pos + n) of a sequence. `tokens[i]` is
  // the id at start_pos + i. When `src_rows` is given and src_rows[i] >= 0,
  // token i is instead read on the GPU from row src_rows[i] of a device token
  // buffer (the previous step's samples, or draft tokens); `tokens` may then
  // be null if every token comes from the device.
  // Returns the first logits row added for this sequence, or -1.
  int add(const int32_t* tokens, int start_pos, int n, const int32_t* src_rows,
          const std::vector<int32_t>& blocks, const SamplingParams& sp, LogitsMode mode);

  // Pads with dummy single-token sequences that write into the scratch KV
  // block 0 (used to reach a CUDA-graph batch-size bucket).
  void pad_decode_to(int num_seqs);

  // Serializes into `dst` (>= max_meta_words() words) and returns the shape.
  BatchShape write(int32_t* dst) const;

  int num_tokens() const { return static_cast<int>(input_ids_.size()); }
  int num_seqs() const { return static_cast<int>(seq_q_start_.size()); }
  int num_logit_rows() const { return static_cast<int>(logit_token_.size()); }
  int max_blocks_per_seq() const { return max_blocks_per_seq_; }
  size_t max_meta_words() const;
  BatchShape shape() const;

 private:
  int max_tokens_, max_seqs_, max_logit_rows_, max_blocks_per_seq_, block_size_;
  int decode_query_threshold_;
  std::vector<int32_t> input_ids_, input_src_, positions_, slot_mapping_;
  std::vector<int32_t> seq_q_start_, seq_q_len_, seq_ctx_len_, block_tables_;
  std::vector<int32_t> logit_token_, param_row_;
  std::vector<float> temperature_, top_p_;
  std::vector<int32_t> top_k_;
  std::vector<uint32_t> seed_lo_, seed_hi_;
};

}  // namespace redline
