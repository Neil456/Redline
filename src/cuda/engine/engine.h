// The Redline inference engine: owns weights, the paged KV cache, the
// scheduler and all device buffers, and runs generation steps.
//
// Execution model
//  * Every step's metadata is serialized into one of two pinned staging
//    buffers and uploaded with a single cudaMemcpyAsync.
//  * Sampling happens on the GPU; sampled ids feed the next step directly on
//    the device, so with async scheduling the host plans and launches step
//    t+1 before it has seen step t's tokens. Only a tiny D2H copy of the
//    sampled ids (into pinned memory) is waited on, one step behind.
//  * Pure-decode steps replay CUDA graphs captured per batch-size bucket.
//  * With a draft model, decode steps run speculative decoding: the draft
//    proposes k tokens, the target verifies them in one forward pass and a
//    GPU kernel performs (lossless) rejection sampling.
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/batch_builder.h"
#include "core/block_allocator.h"
#include "core/json.h"
#include "core/scheduler.h"
#include "core/sequence.h"
#include "cuda/device_memory.h"
#include "cuda/gemm/linear.h"
#include "cuda/model/kv_cache.h"
#include "cuda/model/llama.h"
#include "cuda/model/weights.h"

namespace redline {

struct EngineConfig {
  std::string model_path;
  std::string draft_model_path;  // enables speculative decoding when set
  int num_speculative_tokens = 4;

  int max_num_seqs = 32;
  int max_num_batched_tokens = 2048;
  int max_seq_len = 4096;
  int max_logit_rows = 0;  // 0: derived from max_num_seqs / speculation

  int block_size = 16;
  int64_t kv_cache_tokens = 0;          // KV capacity (capped by the budget); 0 = whole budget
  double gpu_memory_utilization = 0.90;  // device memory fraction the engine may occupy
  KVDType kv_dtype = KVDType::kFP16;

  bool cuda_graphs = true;
  bool async_scheduling = true;
  bool mix_prefill_decode = true;
  LinearOptions linear;
  bool nvtx_per_layer = false;
  bool track_memory = false;  // sample cudaMemGetInfo every step for peak usage
  int device = 0;

  Json to_json() const;
};

struct StepRecord {
  int64_t step = 0;
  std::string kind;  // prefill | decode | mixed | spec
  int num_seqs = 0;
  int num_tokens = 0;
  int num_prefill_tokens = 0;
  int num_decode_tokens = 0;
  int tokens_out = 0;  // tokens appended to sequences
  bool cuda_graph = false;
  double host_ms = 0;  // CPU time to schedule, build and launch the step
  float gpu_ms = 0;    // CUDA-event time of the step on the GPU
  double t_launch = 0, t_done = 0;
};

class Engine {
 public:
  explicit Engine(const EngineConfig& cfg);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Queues a request. arrival_time <= 0 means "now". Returns the request id.
  int64_t add_request(Request req);
  bool has_unfinished() const;

  // Runs one scheduling iteration; returns requests that finished.
  std::vector<RequestOutput> step();

  // Runs the given requests to completion (outputs ordered by id).
  std::vector<RequestOutput> generate(std::vector<Request> requests);

  // Log-probabilities of tokens[i] given tokens[:i] for i >= 1 (teacher
  // forcing, used for perplexity / quality evaluation). Engine must be idle.
  std::vector<float> score(const std::vector<int32_t>& tokens);

  void synchronize();

  const EngineConfig& config() const { return cfg_; }
  const ModelConfig& model_config() const { return weights_->config(); }
  bool speculative() const { return draft_model_ != nullptr; }

  // Memory: ledger categories, device usage and peak.
  Json memory_report();
  Json describe() const;

  void set_record_steps(bool on) { record_steps_ = on; }
  const std::vector<StepRecord>& step_records() const { return records_; }
  void clear_step_records() { records_.clear(); }
  void reset_peak_memory();

 private:
  static constexpr int kStagingSlots = 8;

  struct InFlight {
    int64_t step = 0;
    int slot = 0;
    StepPlan plan;
    std::vector<int> sample_rows;  // per plan entry; -1 when not sampling
    StepRecord record;
  };

  // Where a forward pass writes logits / samples.
  struct ForwardOutputs {
    float* logits = nullptr;   // null: skip lm_head
    int32_t* tokens = nullptr; // null: no sampling
    float* logprobs = nullptr;
    RowStats* stats = nullptr;
    uint32_t rng_stream = 0;
  };

  void init_kv_cache();
  void capture_graphs();
  cudaGraphExec_t capture(LlamaModel& model, const DeviceBatch& b, const int32_t* token_src,
                          const ForwardOutputs& out);
  int graph_bucket(int num_seqs) const;
  ForwardOutputs target_outputs();
  ForwardOutputs draft_outputs(int round);

  InFlight launch_step(StepPlan plan);
  void process_step(InFlight& f, std::vector<RequestOutput>* outputs);
  void run_spec_step(StepPlan& plan, std::vector<RequestOutput>* outputs);

  // Serializes the builder into the next pinned staging slot and enqueues
  // one H2D copy into the device metadata buffer.
  DeviceBatch upload_batch(cudaStream_t s);
  // Resolves device-sourced input ids, runs the model and samples.
  void forward_and_sample(LlamaModel& model, const DeviceBatch& b, const int32_t* token_src,
                          bool static_partitions, const ForwardOutputs& out, cudaStream_t s);

  // Appends a received token; returns true (with finish_reason set) when the
  // sequence is done.
  bool append_token(Sequence* seq, int32_t token, float logprob, double now);
  void finish(Sequence* seq, FinishReason reason, double now, std::vector<RequestOutput>* out);
  void maybe_release(Sequence* seq);
  void sample_memory();

  EngineConfig cfg_;
  int lookahead_ = 0;
  int max_blocks_per_seq_ = 0;
  int max_logit_rows_ = 0;
  int vocab_ = 0;

  cudaStream_t stream_ = nullptr;
  MemoryLedger ledger_;
  std::unique_ptr<LinearExecutor> linear_;
  std::unique_ptr<ModelWeights> weights_, draft_weights_;
  std::unique_ptr<KVCache> kv_, draft_kv_;
  std::unique_ptr<LlamaModel> model_, draft_model_;
  std::unique_ptr<BlockAllocator> allocator_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<BatchBuilder> builder_;

  // Device buffers (allocated once).
  DeviceBuffer meta_dev_;
  DeviceBuffer logits_;            // [max_logit_rows, vocab] fp32
  DeviceBuffer sampled_;           // [max_logit_rows] int32
  DeviceBuffer sampled_logprobs_;  // [max_logit_rows] fp32
  DeviceBuffer target_stats_;      // [max_logit_rows] RowStats
  DeviceBuffer draft_logits_;      // [k, max_num_seqs, vocab] fp32
  DeviceBuffer draft_tokens_;      // [k, max_num_seqs]
  DeviceBuffer draft_stats_;       // [k, max_num_seqs] RowStats
  DeviceBuffer spec_count_;        // [max_num_seqs]
  DeviceBuffer score_targets_;     // [max_logit_rows]
  DeviceBuffer score_out_;         // [max_logit_rows]

  // Pinned host memory: metadata staging ring and per-parity outputs.
  PinnedBuffer meta_host_[kStagingSlots];
  cudaEvent_t ev_h2d_[kStagingSlots]{};
  int staging_cursor_ = 0;
  PinnedBuffer tokens_host_[2];
  PinnedBuffer logprobs_host_[2];
  PinnedBuffer counts_host_;
  cudaEvent_t ev_start_[2]{}, ev_stop_[2]{}, ev_done_[2]{};

  std::vector<int> buckets_;
  std::map<int, cudaGraphExec_t> graphs_;                        // decode, by batch bucket
  std::map<std::pair<int, int>, cudaGraphExec_t> draft_graphs_;  // (bucket, draft round)

  std::unordered_map<int64_t, std::unique_ptr<Sequence>> seqs_;
  std::deque<InFlight> inflight_;
  int64_t next_step_ = 0;

  bool record_steps_ = false;
  std::vector<StepRecord> records_;
  size_t peak_device_used_ = 0;
  size_t init_device_used_ = 0;
};

}  // namespace redline
