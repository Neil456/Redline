// Request / sequence state shared by the scheduler and the engine.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace redline {

struct SamplingParams {
  float temperature = 0.0f;  // <= 0 selects greedy decoding
  int top_k = 0;             // 0 disables top-k
  float top_p = 1.0f;        // 1 disables top-p
  uint64_t seed = 0;
  int max_new_tokens = 128;
  std::vector<int32_t> stop_token_ids;
  bool ignore_eos = false;

  bool greedy() const { return temperature <= 0.0f; }
};

struct Request {
  int64_t id = 0;
  std::vector<int32_t> prompt;
  SamplingParams params;
  double arrival_time = 0.0;  // engine clock (seconds); 0 means "now"
};

enum class SeqStatus { kWaiting, kRunning, kFinished };
enum class FinishReason { kNone, kStop, kLength, kAborted };
const char* finish_reason_name(FinishReason r);

struct Sequence {
  int64_t id = 0;
  std::vector<int32_t> tokens;  // tokens known to the host: prompt + received samples
  int prompt_len = 0;
  SamplingParams params;
  std::vector<int32_t> stop_ids;  // stop tokens incl. EOS unless ignore_eos

  // --- execution state (owned by Scheduler / Engine) ---
  SeqStatus status = SeqStatus::kWaiting;
  FinishReason finish_reason = FinishReason::kNone;
  int num_computed = 0;  // positions whose KV writes have been enqueued on the GPU
  int num_pending = 0;   // tokens sampled on the GPU that the host has not received yet
  int64_t pending_step = -1;  // step that produced the most recent pending token
  int pending_row = -1;       // its row in that step's sample buffer
  bool requeue_when_received = false;  // preempted while a token was in flight
  std::vector<int32_t> blocks;         // block table
  int draft_computed = 0;              // speculative decoding: draft-model KV length

  // --- metrics ---
  double arrival_time = 0.0;
  double first_token_time = -1.0;
  double finish_time = -1.0;
  std::vector<double> token_times;
  std::vector<float> logprobs;
  int64_t draft_proposed = 0;
  int64_t draft_accepted = 0;
  int num_preemptions = 0;

  int total_len() const { return static_cast<int>(tokens.size()) + num_pending; }
  int num_generated() const { return total_len() - prompt_len; }
  int num_uncomputed() const { return total_len() - num_computed; }
  bool is_stop_token(int32_t t) const;
};

struct RequestOutput {
  int64_t id = 0;
  std::vector<int32_t> prompt;
  std::vector<int32_t> output_tokens;
  std::vector<float> logprobs;
  FinishReason finish_reason = FinishReason::kNone;
  double arrival_time = 0.0;
  double first_token_time = 0.0;
  double finish_time = 0.0;
  std::vector<double> token_times;
  int64_t draft_proposed = 0;
  int64_t draft_accepted = 0;
  int num_preemptions = 0;
};

RequestOutput make_output(const Sequence& seq);

}  // namespace redline
