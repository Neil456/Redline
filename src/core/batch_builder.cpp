#include "core/batch_builder.h"

#include <algorithm>
#include <cstring>

#include "core/common.h"

namespace redline {

bool BatchShape::operator==(const BatchShape& o) const {
  return num_tokens == o.num_tokens && num_seqs == o.num_seqs &&
         num_logit_rows == o.num_logit_rows && num_decode_items == o.num_decode_items &&
         num_prefill_seqs == o.num_prefill_seqs && max_prefill_q_tiles == o.max_prefill_q_tiles &&
         max_context == o.max_context;
}

MetaLayout MetaLayout::make(const BatchShape& s, int max_blocks_per_seq) {
  MetaLayout m;
  size_t off = 0;
  auto take = [&](size_t n) {
    const size_t o = off;
    off += n;
    return o;
  };
  const size_t T = s.num_tokens, S = s.num_seqs, R = s.num_logit_rows;
  m.input_ids = take(T);
  m.input_src = take(T);
  m.positions = take(T);
  m.slot_mapping = take(T);
  m.seq_q_start = take(S);
  m.seq_q_len = take(S);
  m.seq_ctx_len = take(S);
  m.block_tables = take(S * static_cast<size_t>(max_blocks_per_seq));
  m.decode_token = take(s.num_decode_items);
  m.decode_seq = take(s.num_decode_items);
  m.prefill_seq = take(s.num_prefill_seqs);
  m.logit_token = take(R);
  m.param_row = take(R);
  m.temperature = take(S);
  m.top_k = take(S);
  m.top_p = take(S);
  m.seed_lo = take(S);
  m.seed_hi = take(S);
  m.total_words = off;
  return m;
}

BatchBuilder::BatchBuilder(int max_tokens, int max_seqs, int max_logit_rows,
                           int max_blocks_per_seq, int block_size, int decode_query_threshold)
    : max_tokens_(max_tokens),
      max_seqs_(max_seqs),
      max_logit_rows_(max_logit_rows),
      max_blocks_per_seq_(max_blocks_per_seq),
      block_size_(block_size),
      decode_query_threshold_(decode_query_threshold) {}

void BatchBuilder::reset() {
  for (auto* v : {&input_ids_, &input_src_, &positions_, &slot_mapping_, &seq_q_start_, &seq_q_len_,
                  &seq_ctx_len_, &block_tables_, &logit_token_, &param_row_, &top_k_}) {
    v->clear();
  }
  temperature_.clear();
  top_p_.clear();
  seed_lo_.clear();
  seed_hi_.clear();
}

size_t BatchBuilder::max_meta_words() const {
  // Upper bound over all shapes: every token could be a decode item and
  // every sequence a prefill sequence.
  BatchShape s;
  s.num_tokens = max_tokens_;
  s.num_seqs = max_seqs_;
  s.num_logit_rows = max_logit_rows_;
  s.num_decode_items = max_tokens_;
  s.num_prefill_seqs = max_seqs_;
  return MetaLayout::make(s, max_blocks_per_seq_).total_words;
}

int BatchBuilder::add(const int32_t* tokens, int start_pos, int n, const int32_t* src_rows,
                      const std::vector<int32_t>& blocks, const SamplingParams& sp,
                      LogitsMode mode) {
  RL_CHECK(n >= 1, "empty sequence segment");
  RL_CHECK(num_tokens() + n <= max_tokens_, "batch exceeds max tokens (", max_tokens_, ")");
  RL_CHECK(num_seqs() + 1 <= max_seqs_, "batch exceeds max sequences (", max_seqs_, ")");
  RL_CHECK(static_cast<int>(blocks.size()) <= max_blocks_per_seq_, "block table too long");
  RL_CHECK(static_cast<int64_t>(blocks.size()) * block_size_ >= start_pos + n,
           "sequence has too few KV blocks for position ", start_pos + n - 1);
  const int seq = num_seqs();
  const int t0 = num_tokens();
  seq_q_start_.push_back(t0);
  seq_q_len_.push_back(n);
  seq_ctx_len_.push_back(start_pos + n);
  block_tables_.insert(block_tables_.end(), blocks.begin(), blocks.end());
  block_tables_.resize(static_cast<size_t>(seq + 1) * max_blocks_per_seq_, 0);
  for (int i = 0; i < n; ++i) {
    const int pos = start_pos + i;
    const bool from_device = src_rows != nullptr && src_rows[i] >= 0;
    RL_CHECK(from_device || tokens != nullptr, "token ", i, " has no source");
    input_ids_.push_back(from_device ? 0 : tokens[i]);
    input_src_.push_back(from_device ? src_rows[i] : -1);
    positions_.push_back(pos);
    slot_mapping_.push_back(blocks[pos / block_size_] * block_size_ + pos % block_size_);
  }
  int first_row = -1;
  if (mode != LogitsMode::kNone) {
    const int rows = mode == LogitsMode::kAll ? n : 1;
    RL_CHECK(num_logit_rows() + rows <= max_logit_rows_, "batch exceeds max logits rows (",
             max_logit_rows_, ")");
    first_row = num_logit_rows();
    for (int i = 0; i < rows; ++i) {
      logit_token_.push_back(mode == LogitsMode::kAll ? t0 + i : t0 + n - 1);
      param_row_.push_back(seq);
    }
  }
  temperature_.push_back(sp.temperature);
  top_k_.push_back(sp.top_k);
  top_p_.push_back(sp.top_p);
  seed_lo_.push_back(static_cast<uint32_t>(sp.seed & 0xffffffffu));
  seed_hi_.push_back(static_cast<uint32_t>(sp.seed >> 32));
  return first_row;
}

void BatchBuilder::pad_decode_to(int target) {
  static const std::vector<int32_t> kScratch = {0};
  const SamplingParams greedy;
  const int32_t tok = 0;
  while (num_seqs() < target) add(&tok, 0, 1, nullptr, kScratch, greedy, LogitsMode::kLast);
}

BatchShape BatchBuilder::shape() const {
  BatchShape s;
  s.num_tokens = num_tokens();
  s.num_seqs = num_seqs();
  s.num_logit_rows = num_logit_rows();
  for (int q = 0; q < num_seqs(); ++q) {
    const int len = seq_q_len_[q];
    if (len <= decode_query_threshold_) {
      s.num_decode_items += len;
    } else {
      s.num_prefill_seqs += 1;
      s.max_prefill_q_tiles =
          std::max(s.max_prefill_q_tiles, (len + kBatchPrefillTileQ - 1) / kBatchPrefillTileQ);
    }
    s.max_context = std::max(s.max_context, seq_ctx_len_[q]);
  }
  return s;
}

BatchShape BatchBuilder::write(int32_t* dst) const {
  const BatchShape s = shape();
  const MetaLayout m = MetaLayout::make(s, max_blocks_per_seq_);
  auto copy_i32 = [&](size_t off, const std::vector<int32_t>& v) {
    if (!v.empty()) std::memcpy(dst + off, v.data(), v.size() * sizeof(int32_t));
  };
  auto copy_4b = [&](size_t off, const void* data, size_t n) {
    if (n) std::memcpy(dst + off, data, n * 4);
  };
  copy_i32(m.input_ids, input_ids_);
  copy_i32(m.input_src, input_src_);
  copy_i32(m.positions, positions_);
  copy_i32(m.slot_mapping, slot_mapping_);
  copy_i32(m.seq_q_start, seq_q_start_);
  copy_i32(m.seq_q_len, seq_q_len_);
  copy_i32(m.seq_ctx_len, seq_ctx_len_);
  copy_i32(m.block_tables, block_tables_);
  size_t nd = 0, np = 0;
  for (int q = 0; q < num_seqs(); ++q) {
    const int len = seq_q_len_[q];
    if (len <= decode_query_threshold_) {
      for (int i = 0; i < len; ++i) {
        dst[m.decode_token + nd] = seq_q_start_[q] + i;
        dst[m.decode_seq + nd] = q;
        ++nd;
      }
    } else {
      dst[m.prefill_seq + np++] = q;
    }
  }
  copy_i32(m.logit_token, logit_token_);
  copy_i32(m.param_row, param_row_);
  copy_4b(m.temperature, temperature_.data(), temperature_.size());
  copy_i32(m.top_k, top_k_);
  copy_4b(m.top_p, top_p_.data(), top_p_.size());
  copy_4b(m.seed_lo, seed_lo_.data(), seed_lo_.size());
  copy_4b(m.seed_hi, seed_hi_.data(), seed_hi_.size());
  return s;
}

}  // namespace redline
