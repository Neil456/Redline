#include "core/scheduler.h"

#include <algorithm>

#include "core/common.h"

namespace redline {

bool StepPlan::pure_decode() const {
  if (seqs.empty()) return false;
  for (const auto& s : seqs) {
    if (s.num_tokens != 1 || !s.sample) return false;
  }
  return true;
}

Scheduler::Scheduler(const SchedulerConfig& cfg, BlockAllocator* allocator)
    : cfg_(cfg), allocator_(allocator) {
  RL_CHECK(cfg_.max_num_seqs > 0 && cfg_.max_num_batched_tokens > 0, "bad scheduler config");
  RL_CHECK(cfg_.max_num_batched_tokens >= cfg_.max_num_seqs,
           "max_num_batched_tokens must be >= max_num_seqs");
}

void Scheduler::add(Sequence* seq) {
  RL_CHECK(seq->prompt_len > 0, "empty prompt");
  RL_CHECK(seq->total_len() + cfg_.lookahead_slots < cfg_.max_seq_len, "prompt of length ",
           seq->total_len(), " does not fit max_seq_len=", cfg_.max_seq_len);
  const int64_t worst_case_tokens = std::min<int64_t>(
      cfg_.max_seq_len, seq->total_len() + seq->params.max_new_tokens + cfg_.lookahead_slots);
  RL_CHECK(ceil_div(worst_case_tokens, cfg_.block_size) <= allocator_->num_total() - 1,
           "request ", seq->id, " needs more KV blocks than the whole cache holds");
  seq->status = SeqStatus::kWaiting;
  waiting_.push_back(seq);
}

bool Scheduler::needs_compute(const Sequence& seq) const {
  if (seq.status == SeqStatus::kFinished) return false;
  if (seq.num_uncomputed() > 1) return true;  // prefill / recompute in progress
  if (seq.num_uncomputed() == 0) return false;
  // One uncomputed token: decode it only if another token is still wanted.
  if (seq.num_generated() >= seq.params.max_new_tokens) return false;
  if (seq.total_len() + cfg_.lookahead_slots >= cfg_.max_seq_len) return false;
  return true;
}

int Scheduler::watermark_blocks() const {
  return static_cast<int>(cfg_.admission_watermark * allocator_->num_total());
}

bool Scheduler::ensure_blocks(Sequence* seq, int num_positions) {
  const int needed = static_cast<int>(ceil_div(num_positions, cfg_.block_size));
  while (static_cast<int>(seq->blocks.size()) < needed) {
    const int b = allocator_->allocate();
    if (b < 0) return false;
    seq->blocks.push_back(b);
  }
  return true;
}

void Scheduler::remove_running(Sequence* seq) {
  auto it = std::find(running_.begin(), running_.end(), seq);
  if (it != running_.end()) running_.erase(it);
}

void Scheduler::preempt(Sequence* seq) {
  RL_DEBUG("preempting sequence ", seq->id, " (", seq->total_len(), " tokens)");
  allocator_->free(seq->blocks);
  seq->blocks.clear();
  seq->num_computed = 0;
  seq->draft_computed = 0;
  seq->status = SeqStatus::kWaiting;
  seq->num_preemptions++;
  remove_running(seq);
  if (seq->num_pending > 0) {
    // Its newest token is still on the GPU; re-admit once it arrives.
    seq->requeue_when_received = true;
    parked_.push_back(seq);
  } else {
    waiting_.push_front(seq);
  }
}

void Scheduler::on_tokens_received(Sequence* seq) {
  if (!seq->requeue_when_received || seq->num_pending > 0) return;
  seq->requeue_when_received = false;
  auto it = std::find(parked_.begin(), parked_.end(), seq);
  if (it != parked_.end()) parked_.erase(it);
  if (seq->status != SeqStatus::kFinished) waiting_.push_front(seq);
}

void Scheduler::finish(Sequence* seq, FinishReason reason) {
  if (seq->status == SeqStatus::kFinished) return;
  seq->status = SeqStatus::kFinished;
  seq->finish_reason = reason;
  allocator_->free(seq->blocks);
  seq->blocks.clear();
  remove_running(seq);
  auto wit = std::find(waiting_.begin(), waiting_.end(), seq);
  if (wit != waiting_.end()) waiting_.erase(wit);
  auto pit = std::find(parked_.begin(), parked_.end(), seq);
  if (pit != parked_.end()) parked_.erase(pit);
}

bool Scheduler::has_unfinished() const {
  return !running_.empty() || !waiting_.empty() || !parked_.empty();
}

StepPlan Scheduler::schedule(int64_t step_id, int64_t prev_step_id) {
  StepPlan plan;
  plan.step_id = step_id;
  int budget = cfg_.max_num_batched_tokens;

  auto add_entry = [&](Sequence* seq, int num_tokens) {
    ScheduledSeq e;
    e.seq = seq;
    e.start_pos = seq->num_computed;
    e.num_tokens = num_tokens;
    e.sample = (seq->num_computed + num_tokens == seq->total_len());
    if (e.start_pos >= static_cast<int>(seq->tokens.size())) {
      // The token at start_pos was sampled by the previous step and is still on the GPU.
      RL_CHECK(num_tokens == 1 && seq->num_pending > 0 && seq->pending_step == prev_step_id,
               "sequence ", seq->id, " needs a device token that is not available");
      e.input_from_device = true;
      e.src_row = seq->pending_row;
    }
    seq->num_computed += num_tokens;
    budget -= num_tokens;
    plan.num_tokens += num_tokens;
    if (num_tokens == 1 && e.sample) {
      plan.num_decode_tokens += 1;
    } else {
      plan.num_prefill_tokens += num_tokens;
    }
    plan.seqs.push_back(e);
  };

  // ---- Prefill candidates (running sequences mid-prefill, then admissions) ----
  auto schedule_prefill = [&]() {
    for (Sequence* seq : std::vector<Sequence*>(running_)) {
      if (budget <= 0) return;
      if (seq->num_uncomputed() <= 1) continue;
      const int chunk = std::min(seq->num_uncomputed(), budget);
      if (!ensure_blocks(seq, seq->num_computed + chunk)) return;
      add_entry(seq, chunk);
    }
    while (!waiting_.empty() && budget > 0 &&
           static_cast<int>(running_.size()) < cfg_.max_num_seqs) {
      Sequence* seq = waiting_.front();
      const int chunk = std::min(seq->num_uncomputed(), budget);
      const int blocks_needed = static_cast<int>(ceil_div(chunk, cfg_.block_size));
      if (allocator_->num_free() - blocks_needed < watermark_blocks() &&
          !running_.empty()) {
        return;  // keep headroom for running sequences
      }
      if (!ensure_blocks(seq, seq->num_computed + chunk)) {
        allocator_->free(seq->blocks);
        seq->blocks.clear();
        return;
      }
      waiting_.pop_front();
      seq->status = SeqStatus::kRunning;
      running_.push_back(seq);
      add_entry(seq, chunk);
    }
  };

  // ---- Decode for running sequences ----
  auto schedule_decode = [&]() {
    for (size_t i = 0; i < running_.size(); ++i) {
      Sequence* seq = running_[i];
      if (budget <= 0) break;
      if (seq->num_uncomputed() != 1 || !needs_compute(*seq)) continue;
      const int positions = seq->num_computed + 1 + cfg_.lookahead_slots;
      bool self_preempted = false;
      while (!ensure_blocks(seq, positions)) {
        // Preempt the newest running sequence that is not yet in this plan.
        Sequence* victim = running_.back();
        const bool victim_scheduled =
            std::any_of(plan.seqs.begin(), plan.seqs.end(),
                        [&](const ScheduledSeq& s) { return s.seq == victim; });
        if (victim == seq || victim_scheduled) {
          preempt(seq);
          self_preempted = true;
          break;
        }
        preempt(victim);
        plan.num_preempted++;
      }
      if (self_preempted) {
        plan.num_preempted++;
        --i;  // running_ shrank at position i
        continue;
      }
      add_entry(seq, 1);
    }
  };

  if (cfg_.mix_prefill_decode) {
    schedule_decode();
    schedule_prefill();
  } else {
    schedule_prefill();
    if (plan.seqs.empty()) schedule_decode();
  }

  // Sequences stuck mid-prefill can exhaust the pool with nothing able to
  // progress; free the newest ones until the oldest can continue.
  while (plan.seqs.empty() && running_.size() > 1) {
    const bool any_work = std::any_of(running_.begin(), running_.end(),
                                      [&](Sequence* s) { return needs_compute(*s); });
    if (!any_work) break;
    preempt(running_.back());
    plan.num_preempted++;
    schedule_prefill();
    if (plan.seqs.empty()) schedule_decode();
  }
  return plan;
}

}  // namespace redline
