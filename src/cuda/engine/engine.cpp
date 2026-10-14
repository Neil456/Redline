#include "cuda/engine/engine.h"

#include <algorithm>
#include <cstring>

#include "core/safetensors.h"
#include "cpu/reference_model.h"
#include "cuda/cuda_utils.h"
#include "cuda/kernels/kernels.h"
#include "cuda/nvtx.h"

namespace redline {
namespace {

constexpr int kMaxSpeculativeTokens = 8;

std::vector<int> make_buckets(int max_seqs) {
  std::vector<int> b;
  for (int v : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256}) {
    if (v < max_seqs) b.push_back(v);
  }
  b.push_back(max_seqs);
  return b;
}

ModelConfig peek_config(const std::string& path) {
  SafetensorsFile f(path);
  return read_model_config(f);
}

}  // namespace

Json EngineConfig::to_json() const {
  Json j = Json::object();
  j.set("model_path", model_path);
  j.set("draft_model_path", draft_model_path);
  j.set("num_speculative_tokens", num_speculative_tokens);
  j.set("max_num_seqs", max_num_seqs);
  j.set("max_num_batched_tokens", max_num_batched_tokens);
  j.set("max_seq_len", max_seq_len);
  j.set("block_size", block_size);
  j.set("kv_cache_tokens", kv_cache_tokens);
  j.set("gpu_memory_utilization", gpu_memory_utilization);
  j.set("kv_dtype", kv_dtype_name(kv_dtype));
  j.set("cuda_graphs", cuda_graphs);
  j.set("async_scheduling", async_scheduling);
  j.set("mix_prefill_decode", mix_prefill_decode);
  j.set("gemv_max_m", linear.gemv_max_m);
  j.set("int8_prefill", prefill_gemm_name(linear.int8_prefill));
  j.set("int4_prefill", prefill_gemm_name(linear.int4_prefill));
  j.set("int4_kernel", int4_kernel);
  return j;
}

Engine::Engine(const EngineConfig& cfg) : cfg_(cfg) {
  RL_NVTX_RANGE("engine_init", kNvtxHost);
  const double t0 = now_seconds();
  RL_CUDA_CHECK(cudaSetDevice(cfg_.device));
  RL_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  const ModelConfig mc = peek_config(cfg_.model_path);
  const bool spec = !cfg_.draft_model_path.empty();
  ModelConfig dc;
  if (spec) {
    dc = peek_config(cfg_.draft_model_path);
    RL_CHECK(dc.vocab_size == mc.vocab_size, "draft vocab (", dc.vocab_size,
             ") must match target vocab (", mc.vocab_size, ")");
    RL_CHECK(cfg_.num_speculative_tokens >= 1 && cfg_.num_speculative_tokens <= kMaxSpeculativeTokens,
             "num_speculative_tokens must be in [1, ", kMaxSpeculativeTokens, "]");
    lookahead_ = cfg_.num_speculative_tokens;
    // Verification needs the accepted count before planning the next step.
    cfg_.async_scheduling = false;
    cfg_.mix_prefill_decode = false;
  }
  RL_CHECK(cfg_.max_seq_len > lookahead_ + 1, "max_seq_len too small");
  if (cfg_.max_seq_len > mc.max_position_embeddings) {
    RL_WARN("max_seq_len ", cfg_.max_seq_len, " exceeds the model's max_position_embeddings ",
            mc.max_position_embeddings);
  }
  const int decode_threshold = lookahead_ + 1;
  const int min_tokens = cfg_.max_num_seqs * decode_threshold;
  if (cfg_.max_num_batched_tokens < min_tokens) {
    RL_WARN("raising max_num_batched_tokens to ", min_tokens);
    cfg_.max_num_batched_tokens = min_tokens;
  }
  max_logit_rows_ = std::max(cfg_.max_logit_rows, std::max(cfg_.max_num_seqs * decode_threshold, 64));
  max_blocks_per_seq_ = static_cast<int>(ceil_div(cfg_.max_seq_len, cfg_.block_size));
  vocab_ = mc.vocab_size;

  // GEMM executor shared by target and draft.
  LinearOptions lo = cfg_.linear;
  lo.max_m = std::max(cfg_.max_num_batched_tokens, max_logit_rows_);
  auto widest = [](const ModelConfig& c, int* k, int* n) {
    *k = std::max({*k, c.hidden_size, c.intermediate_size, c.q_size()});
    *n = std::max({*n, c.qkv_size(), 2 * c.intermediate_size, c.hidden_size});
  };
  widest(mc, &lo.max_k, &lo.max_n);
  if (spec) widest(dc, &lo.max_k, &lo.max_n);
  auto has_int4 = [](const ModelConfig& c) {
    return c.weight_quant == QuantType::kINT4 || c.lm_head_quant == QuantType::kINT4;
  };
  const bool int4_trtllm = lo.int4_prefill == PrefillGemm::kTrtllm;
  constexpr int kInt4GemvMaxRows = 4;
  const int max_decode_rows = cfg_.max_num_seqs * (spec ? cfg_.num_speculative_tokens + 1 : 1);
  RL_CHECK(cfg_.int4_kernel == "auto" || cfg_.int4_kernel == "tc" || cfg_.int4_kernel == "gemv",
           "int4_kernel must be auto, tc or gemv");
  const bool want_tc = cfg_.int4_kernel == "tc" ||
                       (cfg_.int4_kernel == "auto" && max_decode_rows > kInt4GemvMaxRows);
  lo.int4_tensor_cores = want_tc && !int4_trtllm && (has_int4(mc) || (spec && has_int4(dc)));
  if (lo.int4_tensor_cores) {  // an INT4 lm_head also runs on the tensor-core kernel
    if (mc.lm_head_quant == QuantType::kINT4) lo.max_n = std::max(lo.max_n, mc.vocab_size);
    if (spec && dc.lm_head_quant == QuantType::kINT4) lo.max_n = std::max(lo.max_n, dc.vocab_size);
  }
  linear_ = std::make_unique<LinearExecutor>(lo, &ledger_);

  auto trtllm_for = [&](const ModelConfig& c) -> TrtllmWeightOnlyGemm* {
    const bool int4 = c.weight_quant == QuantType::kINT4 &&
                      linear_->resolved_int4_prefill() == PrefillGemm::kTrtllm;
    const bool int8 = c.weight_quant == QuantType::kINT8 &&
                      linear_->resolved_int8_prefill() == PrefillGemm::kTrtllm;
    return (int4 || int8) ? linear_->trtllm() : nullptr;
  };
  const bool tc = linear_->options().int4_tensor_cores;
  weights_ = ModelWeights::load(cfg_.model_path, &ledger_, trtllm_for(mc), tc);
  if (spec) {
    MemoryLedger draft_ledger;
    draft_weights_ = ModelWeights::load(cfg_.draft_model_path, &draft_ledger, trtllm_for(dc), tc);
    ledger_.add("draft_weights", draft_ledger.total());
  }

  ModelRuntimeConfig rc;
  rc.max_tokens = cfg_.max_num_batched_tokens;
  rc.max_logit_rows = max_logit_rows_;
  rc.max_seq_len = cfg_.max_seq_len;
  rc.max_decode_items = cfg_.max_num_seqs * decode_threshold;
  rc.nvtx_per_layer = cfg_.nvtx_per_layer;
  model_ = std::make_unique<LlamaModel>(weights_.get(), linear_.get(), rc, &ledger_, "target");
  if (spec) {
    draft_model_ = std::make_unique<LlamaModel>(draft_weights_.get(), linear_.get(), rc, &ledger_, "draft");
  }

  builder_ = std::make_unique<BatchBuilder>(cfg_.max_num_batched_tokens, cfg_.max_num_seqs,
                                            max_logit_rows_, max_blocks_per_seq_, cfg_.block_size,
                                            decode_threshold);
  const size_t meta_words = builder_->max_meta_words();
  meta_dev_ = ledger_.alloc("activations", meta_words * 4);
  for (int i = 0; i < kStagingSlots; ++i) {
    meta_host_[i] = PinnedBuffer(meta_words * 4);
    RL_CUDA_CHECK(cudaEventCreateWithFlags(&ev_h2d_[i], cudaEventDisableTiming));
  }
  logits_ = ledger_.alloc("activations", static_cast<size_t>(max_logit_rows_) * vocab_ * sizeof(float));
  sampled_ = ledger_.alloc("activations", max_logit_rows_ * sizeof(int32_t));
  sampled_logprobs_ = ledger_.alloc("activations", max_logit_rows_ * sizeof(float));
  target_stats_ = ledger_.alloc("activations", max_logit_rows_ * sizeof(RowStats));
  score_targets_ = ledger_.alloc("activations", max_logit_rows_ * sizeof(int32_t));
  score_out_ = ledger_.alloc("activations", max_logit_rows_ * sizeof(float));
  RL_CUDA_CHECK(cudaMemset(sampled_.get(), 0, sampled_.bytes()));
  if (spec) {
    const size_t rows = static_cast<size_t>(lookahead_) * cfg_.max_num_seqs;
    draft_logits_ = ledger_.alloc("activations", rows * vocab_ * sizeof(float));
    draft_tokens_ = ledger_.alloc("activations", rows * sizeof(int32_t));
    draft_stats_ = ledger_.alloc("activations", rows * sizeof(RowStats));
    spec_count_ = ledger_.alloc("activations", cfg_.max_num_seqs * sizeof(int32_t));
    RL_CUDA_CHECK(cudaMemset(draft_tokens_.get(), 0, draft_tokens_.bytes()));
  }
  for (int i = 0; i < 2; ++i) {
    tokens_host_[i] = PinnedBuffer(max_logit_rows_ * sizeof(int32_t));
    logprobs_host_[i] = PinnedBuffer(max_logit_rows_ * sizeof(float));
    RL_CUDA_CHECK(cudaEventCreate(&ev_start_[i]));
    RL_CUDA_CHECK(cudaEventCreate(&ev_stop_[i]));
    RL_CUDA_CHECK(cudaEventCreateWithFlags(&ev_done_[i], cudaEventDisableTiming));
  }
  counts_host_ = PinnedBuffer(cfg_.max_num_seqs * sizeof(int32_t));

  init_kv_cache();
  model_->bind_kv_cache(kv_.get());
  if (spec) draft_model_->bind_kv_cache(draft_kv_.get());

  SchedulerConfig sc;
  sc.max_num_seqs = cfg_.max_num_seqs;
  sc.max_num_batched_tokens = cfg_.max_num_batched_tokens;
  sc.max_seq_len = cfg_.max_seq_len;
  sc.block_size = cfg_.block_size;
  sc.lookahead_slots = lookahead_;
  sc.mix_prefill_decode = cfg_.mix_prefill_decode;
  scheduler_ = std::make_unique<Scheduler>(sc, allocator_.get());

  RL_CUDA_CHECK(cudaStreamSynchronize(stream_));
  const size_t before_graphs = device_memory_info().used();
  if (cfg_.cuda_graphs) capture_graphs();
  RL_CUDA_CHECK(cudaStreamSynchronize(stream_));
  init_device_used_ = device_memory_info().used();
  ledger_.add("cuda_graphs", init_device_used_ > before_graphs ? init_device_used_ - before_graphs : 0);
  peak_device_used_ = init_device_used_;
  RL_INFO("engine ready in ", static_cast<int>((now_seconds() - t0) * 1000), " ms; ",
          linear_->describe(), "; kv blocks=", allocator_->num_total(), " (",
          format_bytes(kv_->bytes()), ")", spec ? str_cat("; speculative k=", lookahead_) : "",
          "; device used ", format_bytes(init_device_used_));
}

Engine::~Engine() {
  if (stream_) cudaStreamSynchronize(stream_);
  for (auto& kv : graphs_) cudaGraphExecDestroy(kv.second);
  for (auto& kv : draft_graphs_) cudaGraphExecDestroy(kv.second);
  for (int i = 0; i < 2; ++i) {
    if (ev_start_[i]) cudaEventDestroy(ev_start_[i]);
    if (ev_stop_[i]) cudaEventDestroy(ev_stop_[i]);
    if (ev_done_[i]) cudaEventDestroy(ev_done_[i]);
  }
  for (int i = 0; i < kStagingSlots; ++i) {
    if (ev_h2d_[i]) cudaEventDestroy(ev_h2d_[i]);
  }
  if (stream_) cudaStreamDestroy(stream_);
}

void Engine::init_kv_cache() {
  const ModelConfig& mc = weights_->config();
  KVCacheConfig tc;
  tc.num_layers = mc.num_layers;
  tc.num_kv_heads = mc.num_kv_heads;
  tc.head_dim = mc.head_dim;
  tc.block_size = cfg_.block_size;
  tc.dtype = cfg_.kv_dtype;
  KVCacheConfig dcfg = tc;
  if (draft_weights_) {
    const ModelConfig& dc = draft_weights_->config();
    dcfg.num_layers = dc.num_layers;
    dcfg.num_kv_heads = dc.num_kv_heads;
    dcfg.head_dim = dc.head_dim;
  }
  const size_t per_block = tc.bytes_per_block() + (draft_weights_ ? dcfg.bytes_per_block() : 0);
  RL_CUDA_CHECK(cudaDeviceSynchronize());
  const DeviceMemoryInfo info = device_memory_info();
  // Headroom for CUDA graphs, cuBLAS internals and fragmentation.
  const double reserve = static_cast<double>(cfg_.cuda_graphs ? (768u << 20) : (384u << 20));
  const double budget = cfg_.gpu_memory_utilization * static_cast<double>(info.total) -
                        static_cast<double>(info.used()) - reserve;
  const int64_t budget_blocks = budget > 0 ? static_cast<int64_t>(budget / per_block) : 0;
  int64_t num_blocks = budget_blocks;
  if (cfg_.kv_cache_tokens > 0) {
    // Exactly what was asked for (so peak memory reflects the workload),
    // unless that exceeds the memory budget.
    num_blocks = ceil_div(cfg_.kv_cache_tokens, cfg_.block_size) + 1;
    if (num_blocks > budget_blocks) {
      RL_WARN("kv_cache_tokens=", cfg_.kv_cache_tokens, " exceeds the memory budget; capping at ",
              (budget_blocks - 1) * cfg_.block_size, " tokens");
      num_blocks = budget_blocks;
    }
  }
  RL_CHECK(num_blocks >= max_blocks_per_seq_ + 1, "not enough GPU memory for the KV cache: ",
           num_blocks, " blocks, one max_seq_len sequence needs ", max_blocks_per_seq_,
           ". Lower max_seq_len / max_num_batched_tokens or raise gpu_memory_utilization.");
  num_blocks = std::min<int64_t>(num_blocks, 1 << 24);
  tc.num_blocks = dcfg.num_blocks = static_cast<int>(num_blocks);
  kv_ = std::make_unique<KVCache>(tc, &ledger_, "kv_cache");
  if (draft_weights_) draft_kv_ = std::make_unique<KVCache>(dcfg, &ledger_, "draft_kv_cache");
  allocator_ = std::make_unique<BlockAllocator>(static_cast<int>(num_blocks));
}

int Engine::graph_bucket(int num_seqs) const {
  for (int b : buckets_) {
    if (b >= num_seqs) return b;
  }
  return -1;
}

DeviceBatch Engine::upload_batch(cudaStream_t s) {
  const int slot = staging_cursor_;
  staging_cursor_ = (staging_cursor_ + 1) % kStagingSlots;
  // The previous upload from this pinned slot must have left the host.
  RL_CUDA_CHECK(cudaEventSynchronize(ev_h2d_[slot]));
  int32_t* host = meta_host_[slot].as<int32_t>();
  const BatchShape shape = builder_->write(host);
  const size_t words = MetaLayout::make(shape, max_blocks_per_seq_).total_words;
  RL_CUDA_CHECK(cudaMemcpyAsync(meta_dev_.get(), host, words * 4, cudaMemcpyHostToDevice, s));
  RL_CUDA_CHECK(cudaEventRecord(ev_h2d_[slot], s));
  return DeviceBatch::bind(meta_dev_.get(), shape, max_blocks_per_seq_);
}

void Engine::forward_and_sample(LlamaModel& model, const DeviceBatch& b, const int32_t* token_src,
                                bool static_partitions, const ForwardOutputs& out,
                                cudaStream_t s) {
  OpProfiler* prof = op_profiling_ ? &op_profiler_ : nullptr;
  if (prof) prof->begin(&model == model_.get() ? "target" : "draft", s);
  kernels::resolve_input_ids(b.input_ids, b.input_src, token_src, b.shape.num_tokens, s);
  if (prof) prof->mark("input_ids", s);
  model.forward(b, out.logits, static_partitions, s);
  if (out.tokens != nullptr && out.logits != nullptr && b.shape.num_logit_rows > 0) {
    RL_NVTX_RANGE("sample", kNvtxSample);
    kernels::sample_tokens(out.logits, b.shape.num_logit_rows, vocab_, b.sampling, out.rng_stream,
                           out.tokens, out.logprobs, out.stats, s);
    if (prof) prof->mark("sampling", s);
  }
  if (prof) prof->end();
}

void Engine::set_op_profiling(bool on) {
  op_profiling_ = on;
  model_->set_profiler(on ? &op_profiler_ : nullptr);
  if (draft_model_) draft_model_->set_profiler(on ? &op_profiler_ : nullptr);
}

Engine::ForwardOutputs Engine::target_outputs() {
  ForwardOutputs o;
  o.logits = logits_.as<float>();
  o.tokens = sampled_.as<int32_t>();
  o.logprobs = sampled_logprobs_.as<float>();
  o.rng_stream = kRngSample;
  return o;
}

Engine::ForwardOutputs Engine::draft_outputs(int round) {
  const size_t row0 = static_cast<size_t>(round) * cfg_.max_num_seqs;
  ForwardOutputs o;
  o.logits = draft_logits_.as<float>() + row0 * vocab_;
  o.tokens = draft_tokens_.as<int32_t>() + row0;
  o.stats = draft_stats_.as<RowStats>() + row0;
  o.rng_stream = kRngDraft;
  return o;
}

cudaGraphExec_t Engine::capture(LlamaModel& model, const DeviceBatch& b, const int32_t* token_src,
                                const ForwardOutputs& out) {
  // Warm up eagerly first so lazy module loading and cuBLASLt plan creation
  // happen outside the capture.
  forward_and_sample(model, b, token_src, true, out, stream_);
  RL_CUDA_CHECK(cudaStreamSynchronize(stream_));
  cudaGraph_t graph;
  RL_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  forward_and_sample(model, b, token_src, true, out, stream_);
  RL_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  cudaGraphExec_t exec;
  RL_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  RL_CUDA_CHECK(cudaGraphDestroy(graph));
  return exec;
}

void Engine::capture_graphs() {
  RL_NVTX_RANGE("capture_graphs", kNvtxHost);
  const double t0 = now_seconds();
  buckets_ = make_buckets(cfg_.max_num_seqs);
  for (int b : buckets_) {
    builder_->reset();
    builder_->pad_decode_to(b);
    const DeviceBatch batch = upload_batch(stream_);
    if (draft_model_) {
      // Speculative mode: target steps are verification passes (eager);
      // the draft's single-token rounds are graph-captured per round.
      for (int j = 0; j < lookahead_; ++j) {
        draft_graphs_[{b, j}] = capture(*draft_model_, batch, draft_tokens_.as<int32_t>(), draft_outputs(j));
      }
    } else {
      graphs_[b] = capture(*model_, batch, sampled_.as<int32_t>(), target_outputs());
    }
  }
  RL_INFO("captured CUDA graphs for batch sizes up to ", buckets_.back(), " in ",
          static_cast<int>((now_seconds() - t0) * 1000), " ms");
}

int64_t Engine::add_request(Request req) {
  RL_CHECK(!req.prompt.empty(), "empty prompt");
  RL_CHECK(seqs_.count(req.id) == 0, "duplicate request id ", req.id);
  for (int32_t t : req.prompt) RL_CHECK(t >= 0 && t < vocab_, "token id ", t, " out of range");
  RL_CHECK(req.params.max_new_tokens >= 1, "max_new_tokens must be >= 1");
  auto seq = std::make_unique<Sequence>();
  seq->id = req.id;
  seq->tokens = std::move(req.prompt);
  seq->prompt_len = static_cast<int>(seq->tokens.size());
  seq->params = req.params;
  seq->stop_ids = req.params.stop_token_ids;
  if (!req.params.ignore_eos) {
    for (int e : weights_->config().eos_token_ids) seq->stop_ids.push_back(e);
  }
  seq->arrival_time = req.arrival_time > 0 ? req.arrival_time : now_seconds();
  scheduler_->add(seq.get());
  seqs_[req.id] = std::move(seq);
  return req.id;
}

bool Engine::has_unfinished() const { return scheduler_->has_unfinished() || !inflight_.empty(); }

std::vector<RequestOutput> Engine::step() {
  RL_NVTX_RANGE("step", kNvtxStep);
  std::vector<RequestOutput> outputs;
  if (draft_model_) {
    StepPlan plan = scheduler_->schedule(next_step_, -1);
    if (plan.empty()) {
      RL_CHECK(!scheduler_->has_unfinished(), "scheduler made no progress");
      return outputs;
    }
    if (plan.pure_decode()) {
      run_spec_step(plan, &outputs);
    } else {
      InFlight f = launch_step(std::move(plan));
      process_step(f, &outputs);
    }
    return outputs;
  }

  const int64_t prev = inflight_.empty() ? -1 : inflight_.back().step;
  StepPlan plan;
  {
    RL_NVTX_RANGE("schedule", kNvtxHost);
    plan = scheduler_->schedule(next_step_, prev);
  }
  const bool launched = !plan.empty();
  if (launched) inflight_.push_back(launch_step(std::move(plan)));
  // With async scheduling, keep the step we just launched in flight and
  // retire the previous one: the GPU never waits for the host.
  const size_t keep = (cfg_.async_scheduling && launched) ? 1 : 0;
  while (inflight_.size() > keep) {
    process_step(inflight_.front(), &outputs);
    inflight_.pop_front();
  }
  RL_CHECK(launched || !inflight_.empty() || !scheduler_->has_unfinished() || !outputs.empty(),
           "scheduler made no progress");
  return outputs;
}

Engine::InFlight Engine::launch_step(StepPlan plan) {
  RL_NVTX_RANGE("launch", kNvtxHost);
  const double t_start = now_seconds();
  InFlight f;
  f.step = next_step_++;
  f.slot = static_cast<int>(f.step % 2);
  f.plan = std::move(plan);
  const StepPlan& p = f.plan;

  builder_->reset();
  for (const ScheduledSeq& e : p.seqs) {
    Sequence* seq = e.seq;
    const int32_t* toks = e.input_from_device ? nullptr : seq->tokens.data() + e.start_pos;
    const int32_t src = e.src_row;
    const int row = builder_->add(toks, e.start_pos, e.num_tokens,
                                  e.input_from_device ? &src : nullptr, seq->blocks, seq->params,
                                  e.sample ? LogitsMode::kLast : LogitsMode::kNone);
    f.sample_rows.push_back(e.sample ? row : -1);
    if (e.sample) {
      seq->num_pending++;
      seq->pending_step = f.step;
      seq->pending_row = row;
    }
    if (draft_model_) seq->draft_computed = e.start_pos + e.num_tokens;
  }
  const int real_seqs = builder_->num_seqs();
  const int bucket = (!graphs_.empty() && !op_profiling_ && p.pure_decode()) ? graph_bucket(real_seqs) : -1;
  if (bucket > 0) builder_->pad_decode_to(bucket);

  RL_CUDA_CHECK(cudaEventRecord(ev_start_[f.slot], stream_));
  const DeviceBatch b = upload_batch(stream_);
  if (bucket > 0) {
    RL_CUDA_CHECK(cudaGraphLaunch(graphs_.at(bucket), stream_));
  } else {
    forward_and_sample(*model_, b, sampled_.as<int32_t>(), false, target_outputs(), stream_);
    if (draft_model_) {
      // Keep the draft model's KV cache in sync with the target's prefill.
      ForwardOutputs none;
      forward_and_sample(*draft_model_, b, sampled_.as<int32_t>(), false, none, stream_);
    }
  }
  RL_CUDA_CHECK(cudaEventRecord(ev_stop_[f.slot], stream_));
  const int rows = b.shape.num_logit_rows;
  if (rows > 0) {
    RL_CUDA_CHECK(cudaMemcpyAsync(tokens_host_[f.slot].get(), sampled_.get(), rows * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, stream_));
    RL_CUDA_CHECK(cudaMemcpyAsync(logprobs_host_[f.slot].get(), sampled_logprobs_.get(),
                                  rows * sizeof(float), cudaMemcpyDeviceToHost, stream_));
  }
  RL_CUDA_CHECK(cudaEventRecord(ev_done_[f.slot], stream_));

  StepRecord& r = f.record;
  r.step = f.step;
  r.num_seqs = real_seqs;
  r.num_tokens = p.num_tokens;
  r.num_prefill_tokens = p.num_prefill_tokens;
  r.num_decode_tokens = p.num_decode_tokens;
  r.kind = p.num_prefill_tokens == 0 ? "decode" : (p.num_decode_tokens == 0 ? "prefill" : "mixed");
  r.cuda_graph = bucket > 0;
  record_engine_state(&r);
  r.t_launch = now_seconds();
  r.host_ms = (r.t_launch - t_start) * 1000.0;
  return f;
}

void Engine::record_engine_state(StepRecord* r) const {
  r->kv_blocks_used = allocator_->num_used();
  r->kv_blocks_total = allocator_->num_total() - 1;  // block 0 is the scratch block
  r->num_running = scheduler_->num_running();
  r->num_waiting = scheduler_->num_waiting();
}

void Engine::process_step(InFlight& f, std::vector<RequestOutput>* outputs) {
  RL_NVTX_RANGE("process", kNvtxHost);
  RL_CUDA_CHECK(cudaEventSynchronize(ev_done_[f.slot]));
  const double now = now_seconds();
  const int32_t* tokens = tokens_host_[f.slot].as<int32_t>();
  const float* lps = logprobs_host_[f.slot].as<float>();
  int appended = 0;
  for (size_t i = 0; i < f.plan.seqs.size(); ++i) {
    const int row = f.sample_rows[i];
    if (row < 0) continue;
    Sequence* seq = f.plan.seqs[i].seq;
    seq->num_pending--;
    if (seq->status == SeqStatus::kFinished) {  // stopped earlier; drop the extra token
      maybe_release(seq);
      continue;
    }
    ++appended;
    if (append_token(seq, tokens[row], lps[row], now)) {
      finish(seq, seq->finish_reason, now, outputs);
    } else {
      scheduler_->on_tokens_received(seq);
    }
  }
  if (record_steps_) {
    StepRecord r = f.record;
    RL_CUDA_CHECK(cudaEventElapsedTime(&r.gpu_ms, ev_start_[f.slot], ev_stop_[f.slot]));
    r.tokens_out = appended;
    r.t_done = now;
    records_.push_back(r);
  }
  sample_memory();
}

bool Engine::append_token(Sequence* seq, int32_t token, float logprob, double now) {
  seq->tokens.push_back(token);
  seq->logprobs.push_back(logprob);
  seq->token_times.push_back(now);
  if (seq->first_token_time < 0) seq->first_token_time = now;
  const int generated = static_cast<int>(seq->tokens.size()) - seq->prompt_len;
  if (seq->is_stop_token(token)) {
    seq->finish_reason = FinishReason::kStop;
    return true;
  }
  if (generated >= seq->params.max_new_tokens ||
      static_cast<int>(seq->tokens.size()) + lookahead_ >= cfg_.max_seq_len) {
    seq->finish_reason = FinishReason::kLength;
    return true;
  }
  return false;
}

void Engine::finish(Sequence* seq, FinishReason reason, double now,
                    std::vector<RequestOutput>* outputs) {
  seq->finish_time = now;
  scheduler_->finish(seq, reason);
  outputs->push_back(make_output(*seq));
  maybe_release(seq);
}

void Engine::maybe_release(Sequence* seq) {
  if (seq->status == SeqStatus::kFinished && seq->num_pending == 0) seqs_.erase(seq->id);
}

void Engine::run_spec_step(StepPlan& plan, std::vector<RequestOutput>* outputs) {
  RL_NVTX_RANGE("spec_step", kNvtxSpec);
  const double t_start = now_seconds();
  const int B = static_cast<int>(plan.seqs.size());
  const int K = lookahead_;
  const int S = cfg_.max_num_seqs;  // row stride of the draft buffers
  const int64_t step_id = next_step_++;
  std::vector<int> n_before(B);
  for (int i = 0; i < B; ++i) n_before[i] = plan.seqs[i].seq->total_len();
  const int bucket = (draft_graphs_.empty() || op_profiling_) ? -1 : graph_bucket(B);

  RL_CUDA_CHECK(cudaEventRecord(ev_start_[0], stream_));
  // Draft round 0 catches the draft KV cache up (1 or 2 tokens per sequence)
  // and proposes d_1; rounds 1..K-1 propose d_2..d_K from device tokens.
  {
    RL_NVTX_RANGE("draft", kNvtxSpec);
    for (int j = 0; j < K; ++j) {
      builder_->reset();
      bool single = true;
      for (int i = 0; i < B; ++i) {
        Sequence* seq = plan.seqs[i].seq;
        if (j == 0) {
          const int start = seq->draft_computed;
          const int n = n_before[i] - start;
          single = single && n == 1;
          builder_->add(seq->tokens.data() + start, start, n, nullptr, seq->blocks, seq->params,
                        LogitsMode::kLast);
        } else {
          const int32_t src = (j - 1) * S + i;
          builder_->add(nullptr, n_before[i] - 1 + j, 1, &src, seq->blocks, seq->params,
                        LogitsMode::kLast);
        }
      }
      if (bucket > 0 && single) {
        builder_->pad_decode_to(bucket);
        upload_batch(stream_);
        RL_CUDA_CHECK(cudaGraphLaunch(draft_graphs_.at({bucket, j}), stream_));
      } else {
        const DeviceBatch b = upload_batch(stream_);
        forward_and_sample(*draft_model_, b, draft_tokens_.as<int32_t>(), false, draft_outputs(j),
                           stream_);
      }
    }
  }
  // Target verification: [t_{n-1}, d_1..d_K] at positions n-1..n-1+K.
  {
    RL_NVTX_RANGE("verify", kNvtxSpec);
    builder_->reset();
    std::vector<int32_t> toks(K + 1, 0), src(K + 1, -1);
    for (int i = 0; i < B; ++i) {
      Sequence* seq = plan.seqs[i].seq;
      toks[0] = seq->tokens[n_before[i] - 1];
      for (int j = 1; j <= K; ++j) src[j] = (j - 1) * S + i;
      builder_->add(toks.data(), n_before[i] - 1, K + 1, src.data(), seq->blocks, seq->params,
                    LogitsMode::kAll);
    }
    const DeviceBatch b = upload_batch(stream_);
    kernels::resolve_input_ids(b.input_ids, b.input_src, draft_tokens_.as<int32_t>(),
                               b.shape.num_tokens, stream_);
    model_->forward(b, logits_.as<float>(), false, stream_);
    const int R = b.shape.num_logit_rows;
    kernels::compute_row_stats(logits_.as<float>(), R, vocab_, b.sampling,
                               target_stats_.as<RowStats>(), stream_);
    kernels::speculative_verify(logits_.as<float>(), target_stats_.as<RowStats>(),
                                draft_logits_.as<float>(), draft_stats_.as<RowStats>(),
                                draft_tokens_.as<int32_t>(), B, K, S, vocab_, b.sampling,
                                sampled_.as<int32_t>(), sampled_logprobs_.as<float>(),
                                spec_count_.as<int32_t>(), stream_);
  }
  RL_CUDA_CHECK(cudaEventRecord(ev_stop_[0], stream_));
  const int R = B * (K + 1);
  RL_CUDA_CHECK(cudaMemcpyAsync(tokens_host_[0].get(), sampled_.get(), R * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream_));
  RL_CUDA_CHECK(cudaMemcpyAsync(logprobs_host_[0].get(), sampled_logprobs_.get(), R * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_));
  RL_CUDA_CHECK(cudaMemcpyAsync(counts_host_.get(), spec_count_.get(), B * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream_));
  RL_CUDA_CHECK(cudaEventRecord(ev_done_[0], stream_));
  const double t_launch = now_seconds();
  RL_CUDA_CHECK(cudaEventSynchronize(ev_done_[0]));
  const double now = now_seconds();

  const int32_t* tokens = tokens_host_[0].as<int32_t>();
  const float* lps = logprobs_host_[0].as<float>();
  const int32_t* counts = counts_host_.as<int32_t>();
  int appended = 0, step_accepted = 0;
  for (int i = 0; i < B; ++i) {
    Sequence* seq = plan.seqs[i].seq;
    const int n = n_before[i];
    const int count = counts[i];
    const int accepted = count - 1;
    step_accepted += accepted;
    seq->draft_proposed += K;
    seq->draft_accepted += accepted;
    seq->num_computed = n + accepted;  // target KV valid for positions [0, n - 1 + accepted]
    seq->draft_computed = std::min(n + accepted, n + K - 1);
    for (int t = 0; t < count; ++t) {
      ++appended;
      const int row = i * (K + 1) + t;
      if (append_token(seq, tokens[row], lps[row], now)) {
        finish(seq, seq->finish_reason, now, outputs);
        break;
      }
    }
  }
  if (record_steps_) {
    StepRecord r;
    r.step = step_id;
    r.kind = "spec";
    r.num_seqs = B;
    r.num_tokens = B * (K + 1);
    r.num_decode_tokens = B;
    r.tokens_out = appended;
    r.cuda_graph = bucket > 0;
    r.draft_proposed = B * K;
    r.draft_accepted = step_accepted;
    record_engine_state(&r);
    r.host_ms = (t_launch - t_start) * 1000.0;
    RL_CUDA_CHECK(cudaEventElapsedTime(&r.gpu_ms, ev_start_[0], ev_stop_[0]));
    r.t_launch = t_launch;
    r.t_done = now;
    records_.push_back(r);
  }
  sample_memory();
}

std::vector<RequestOutput> Engine::generate(std::vector<Request> requests) {
  for (auto& r : requests) add_request(std::move(r));
  std::vector<RequestOutput> outputs;
  while (has_unfinished()) {
    auto out = step();
    for (auto& o : out) outputs.push_back(std::move(o));
  }
  std::sort(outputs.begin(), outputs.end(),
            [](const RequestOutput& a, const RequestOutput& b) { return a.id < b.id; });
  return outputs;
}

std::vector<float> Engine::score(const std::vector<int32_t>& tokens) {
  RL_NVTX_RANGE("score", kNvtxPrefill);
  RL_CHECK(!has_unfinished(), "score() requires an idle engine");
  const int L = static_cast<int>(tokens.size());
  RL_CHECK(L >= 2 && L <= cfg_.max_seq_len, "score: sequence length must be in [2, max_seq_len]");
  for (int32_t t : tokens) RL_CHECK(t >= 0 && t < vocab_, "token id out of range");
  std::vector<int32_t> blocks;
  for (int64_t i = 0; i < ceil_div(L, cfg_.block_size); ++i) {
    const int blk = allocator_->allocate();
    if (blk < 0) {
      allocator_->free(blocks);
      throw Error("score: KV cache too small for the sequence");
    }
    blocks.push_back(blk);
  }
  std::vector<float> out(L - 1);
  const int chunk = std::min(cfg_.max_num_batched_tokens, max_logit_rows_);
  const SamplingParams greedy;
  int32_t* targets_host = tokens_host_[0].as<int32_t>();
  float* lp_host = logprobs_host_[0].as<float>();
  for (int pos = 0; pos < L; pos += chunk) {
    const int n = std::min(chunk, L - pos);
    builder_->reset();
    builder_->add(tokens.data() + pos, pos, n, nullptr, blocks, greedy, LogitsMode::kAll);
    const DeviceBatch b = upload_batch(stream_);
    ForwardOutputs o;
    o.logits = logits_.as<float>();
    forward_and_sample(*model_, b, sampled_.as<int32_t>(), false, o, stream_);
    for (int i = 0; i < n; ++i) targets_host[i] = pos + i + 1 < L ? tokens[pos + i + 1] : 0;
    RL_CUDA_CHECK(cudaMemcpyAsync(score_targets_.get(), targets_host, n * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream_));
    kernels::gather_logprobs(logits_.as<float>(), score_targets_.as<int32_t>(), n, vocab_,
                             score_out_.as<float>(), stream_);
    RL_CUDA_CHECK(cudaMemcpyAsync(lp_host, score_out_.get(), n * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_));
    RL_CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (int i = 0; i < n && pos + i + 1 < L; ++i) out[pos + i] = lp_host[i];
  }
  allocator_->free(blocks);
  return out;
}

void Engine::synchronize() { RL_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

void Engine::sample_memory() {
  if (!cfg_.track_memory) return;
  peak_device_used_ = std::max(peak_device_used_, device_memory_info().used());
}

void Engine::reset_peak_memory() { peak_device_used_ = device_memory_info().used(); }

Json Engine::memory_report() {
  const DeviceMemoryInfo info = device_memory_info();
  peak_device_used_ = std::max(peak_device_used_, info.used());
  Json j = Json::object();
  j.set("ledger_bytes", ledger_.to_json());
  j.set("device_used_bytes", static_cast<uint64_t>(info.used()));
  j.set("device_total_bytes", static_cast<uint64_t>(info.total));
  j.set("device_used_after_init_bytes", static_cast<uint64_t>(init_device_used_));
  j.set("peak_device_used_bytes", static_cast<uint64_t>(peak_device_used_));
  j.set("kv_cache_blocks", allocator_->num_total());
  j.set("kv_cache_token_capacity", static_cast<int64_t>(allocator_->num_total() - 1) * cfg_.block_size);
  return j;
}

Json Engine::describe() const {
  Json j = Json::object();
  j.set("engine", cfg_.to_json());
  j.set("model", weights_->config().to_json());
  if (draft_weights_) j.set("draft_model", draft_weights_->config().to_json());
  j.set("linear", linear_->describe());
  j.set("graph_buckets", Json::array_of(buckets_));
  return j;
}

}  // namespace redline
