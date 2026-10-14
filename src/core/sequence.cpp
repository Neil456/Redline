#include "core/sequence.h"

#include <algorithm>

namespace redline {

const char* finish_reason_name(FinishReason r) {
  switch (r) {
    case FinishReason::kNone: return "none";
    case FinishReason::kStop: return "stop";
    case FinishReason::kLength: return "length";
    case FinishReason::kAborted: return "aborted";
  }
  return "?";
}

bool Sequence::is_stop_token(int32_t t) const {
  return std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end();
}

RequestOutput make_output(const Sequence& seq) {
  RequestOutput out;
  out.id = seq.id;
  out.prompt.assign(seq.tokens.begin(), seq.tokens.begin() + seq.prompt_len);
  out.output_tokens.assign(seq.tokens.begin() + seq.prompt_len, seq.tokens.end());
  out.logprobs = seq.logprobs;
  out.finish_reason = seq.finish_reason;
  out.arrival_time = seq.arrival_time;
  out.first_token_time = seq.first_token_time;
  out.finish_time = seq.finish_time;
  out.token_times = seq.token_times;
  out.draft_proposed = seq.draft_proposed;
  out.draft_accepted = seq.draft_accepted;
  out.num_preemptions = seq.num_preemptions;
  return out;
}

}  // namespace redline
