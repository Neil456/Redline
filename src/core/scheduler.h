// Continuous-batching scheduler. Each call to schedule() decides which
// sequences run in the next forward step and how many tokens each processes:
// decode steps for running sequences first (one token each, FCFS), then
// chunked prefill for partially-prefilled and newly admitted sequences within
// the per-step token budget. When the KV pool runs out during decode the
// most recently admitted sequence is preempted and later recomputed.
//
// The scheduler understands asynchronous execution: a sequence's latest token
// may still be on the GPU ("pending"); its next decode step then reads the
// token straight from the previous step's sample buffer.
#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "core/block_allocator.h"
#include "core/sequence.h"

namespace redline {

struct SchedulerConfig {
  int max_num_seqs = 64;
  int max_num_batched_tokens = 2048;
  int max_seq_len = 4096;
  int block_size = 16;
  // Extra KV slots a decode step may write past the next position
  // (speculative decoding writes num_speculative_tokens extra positions).
  int lookahead_slots = 0;
  // Allow prefill chunks and decode tokens in the same step. When false,
  // prefill-only steps take priority over decode steps.
  bool mix_prefill_decode = true;
  // Fraction of the KV pool kept free when admitting new sequences, so
  // running sequences can keep decoding without immediate preemption.
  double admission_watermark = 0.01;
};

struct ScheduledSeq {
  Sequence* seq = nullptr;
  int start_pos = 0;   // first position processed this step
  int num_tokens = 0;  // positions [start_pos, start_pos + num_tokens)
  bool sample = false; // sample a new token after the last position
  bool input_from_device = false;  // token at start_pos is a pending GPU sample
  int src_row = -1;                // row in the previous step's sample buffer
};

struct StepPlan {
  int64_t step_id = 0;
  std::vector<ScheduledSeq> seqs;
  int num_tokens = 0;
  int num_prefill_tokens = 0;
  int num_decode_tokens = 0;
  int num_preempted = 0;

  bool empty() const { return seqs.empty(); }
  // Every sequence processes exactly one token and samples (CUDA-graph eligible).
  bool pure_decode() const;
};

class Scheduler {
 public:
  Scheduler(const SchedulerConfig& cfg, BlockAllocator* allocator);

  const SchedulerConfig& config() const { return cfg_; }

  // Takes a non-owning pointer; the engine owns sequence storage.
  void add(Sequence* seq);

  // Plans step `step_id`. `prev_step_id` is the step whose samples have not
  // been received by the host yet (or -1).
  StepPlan schedule(int64_t step_id, int64_t prev_step_id);

  // Marks a sequence finished and releases its KV blocks.
  void finish(Sequence* seq, FinishReason reason);

  // A preempted sequence's in-flight token arrived; it can be re-admitted.
  void on_tokens_received(Sequence* seq);

  bool has_unfinished() const;
  int num_running() const { return static_cast<int>(running_.size()); }
  int num_waiting() const { return static_cast<int>(waiting_.size() + parked_.size()); }
  const std::vector<Sequence*>& running() const { return running_; }

  // True if the sequence still needs GPU work (not at its generation limit).
  bool needs_compute(const Sequence& seq) const;

 private:
  bool ensure_blocks(Sequence* seq, int num_positions);
  void preempt(Sequence* seq);
  void remove_running(Sequence* seq);
  int watermark_blocks() const;

  SchedulerConfig cfg_;
  BlockAllocator* allocator_;
  std::vector<Sequence*> running_;  // admission order (oldest first)
  std::deque<Sequence*> waiting_;
  std::vector<Sequence*> parked_;   // preempted, waiting for in-flight tokens
};

}  // namespace redline
