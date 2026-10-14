// End-to-end engine tests on small random models (generated in C++), using
// the fp32 CPU reference model as the oracle.
//
// Greedy outputs are validated with a robust criterion: every generated token
// must be (near-)argmax of the reference model's logits given the same prefix.
// That tolerates legitimate fp16 rounding differences on near-ties while
// catching any real bug (wrong KV slot, RoPE, masking, sampling, ...).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>

#include "core/common.h"
#include "cpu/random_model.h"
#include "cpu/reference_model.h"
#include "cuda/bench/harness.h"
#include "cuda/engine/engine.h"
#include "test_framework.h"

using namespace redline;

namespace {

std::string tmp(const std::string& name) {
  const char* dir = std::getenv("TMPDIR");
  return std::string(dir ? dir : "/tmp") + "/redline_gpu_" + name + ".rlm";
}

const std::string& model_path(QuantType q, uint64_t seed = 1) {
  static std::map<std::pair<int, uint64_t>, std::string> cache;
  auto key = std::make_pair(static_cast<int>(q), seed);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  const std::string path = tmp(std::string(quant_type_name(q)) + "_" + std::to_string(seed));
  write_random_model(path, tiny_test_config(q), seed);
  return cache[key] = path;
}

CpuReferenceModel& reference(const std::string& path) {
  static std::map<std::string, std::unique_ptr<CpuReferenceModel>> refs;
  auto& r = refs[path];
  if (!r) r = std::make_unique<CpuReferenceModel>(path);
  return *r;
}

EngineConfig small_config(const std::string& path) {
  EngineConfig c;
  c.model_path = path;
  c.max_num_seqs = 8;
  c.max_num_batched_tokens = 256;
  c.max_seq_len = 512;
  c.kv_cache_tokens = 4096;
  return c;
}

std::vector<Request> make_requests(int n, int prompt_len, int new_tokens, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<Request> reqs;
  for (int i = 0; i < n; ++i) {
    Request r;
    r.id = i;
    const int len = prompt_len + static_cast<int>(rng() % 7);
    for (int t = 0; t < len; ++t) r.prompt.push_back(3 + static_cast<int32_t>(rng() % 500));
    r.params.max_new_tokens = new_tokens;
    r.params.ignore_eos = true;
    reqs.push_back(std::move(r));
  }
  return reqs;
}

// Every generated token must be within `margin` of the reference argmax logit.
void check_greedy(const std::string& path, const std::vector<RequestOutput>& outs, float margin) {
  CpuReferenceModel& ref = reference(path);
  const int V = ref.config().vocab_size;
  for (const auto& o : outs) {
    std::vector<int32_t> seq = o.prompt;
    seq.insert(seq.end(), o.output_tokens.begin(), o.output_tokens.end());
    seq.pop_back();  // logits for the last generated token's prefix suffice
    const auto logits = ref.forward(seq);
    for (size_t i = 0; i < o.output_tokens.size(); ++i) {
      const float* row = logits.data() + (o.prompt.size() - 1 + i) * V;
      const float best = *std::max_element(row, row + V);
      const float got = row[o.output_tokens[i]];
      if (!(got >= best - margin)) {
        throw rltest::Failure{str_cat("request ", o.id, " token ", i, ": logit ", got,
                                      " vs reference max ", best)};
      }
    }
  }
}

}  // namespace

TEST(engine_score_matches_reference) {
  for (QuantType q : {QuantType::kFP16, QuantType::kINT8, QuantType::kINT4}) {
    const std::string& path = model_path(q);
    EngineConfig cfg = small_config(path);
    cfg.max_logit_rows = 64;  // forces several scoring chunks
    Engine engine(cfg);
    std::vector<int32_t> toks;
    for (int i = 0; i < 150; ++i) toks.push_back(3 + (i * 37) % 500);
    const auto lp = engine.score(toks);
    const auto logits = reference(path).forward(toks);
    const int V = engine.model_config().vocab_size;
    double max_err = 0, mean_err = 0;
    for (size_t i = 0; i + 1 < toks.size(); ++i) {
      const float* row = logits.data() + i * V;
      const double mx = *std::max_element(row, row + V);
      double s = 0;
      for (int v = 0; v < V; ++v) s += std::exp(row[v] - mx);
      const double ref_lp = row[toks[i + 1]] - mx - std::log(s);
      const double e = std::fabs(ref_lp - lp[i]);
      max_err = std::max(max_err, e);
      mean_err += e / (toks.size() - 1);
    }
    std::printf("    %s: logprob error mean %.4f max %.4f\n", quant_type_name(q), mean_err, max_err);
    EXPECT_TRUE(mean_err < 0.03);
    EXPECT_TRUE(max_err < 0.3);
  }
}

TEST(engine_greedy_matches_reference_all_modes) {
  const std::string& path = model_path(QuantType::kFP16);
  struct Mode {
    bool graphs, async, mixed;
  };
  for (Mode m : {Mode{true, true, true}, Mode{false, false, true}, Mode{true, false, false}, Mode{false, true, false}}) {
    EngineConfig cfg = small_config(path);
    cfg.cuda_graphs = m.graphs;
    cfg.async_scheduling = m.async;
    cfg.mix_prefill_decode = m.mixed;
    cfg.max_num_batched_tokens = 64;  // chunked prefill
    Engine engine(cfg);
    const auto outs = engine.generate(make_requests(6, 40, 24, 5));
    EXPECT_EQ(outs.size(), static_cast<size_t>(6));
    for (const auto& o : outs) {
      EXPECT_EQ(o.output_tokens.size(), static_cast<size_t>(24));
      EXPECT_TRUE(o.finish_reason == FinishReason::kLength);
    }
    check_greedy(path, outs, 0.1f);
  }
}

TEST(engine_quantized_models_and_int8_kv) {
  for (QuantType q : {QuantType::kINT8, QuantType::kINT4}) {
    const std::string& path = model_path(q);
    Engine engine(small_config(path));
    check_greedy(path, engine.generate(make_requests(4, 30, 16, 7)), 0.15f);
  }
  const std::string& path = model_path(QuantType::kFP16);
  EngineConfig cfg = small_config(path);
  cfg.kv_dtype = KVDType::kINT8;
  Engine engine(cfg);
  check_greedy(path, engine.generate(make_requests(4, 30, 16, 8)), 0.5f);
}

TEST(engine_preemption_recompute) {
  const std::string& path = model_path(QuantType::kFP16);
  EngineConfig cfg = small_config(path);
  cfg.kv_cache_tokens = 160;  // 4 x (~45 + 40) tokens cannot all fit
  cfg.max_num_seqs = 4;
  cfg.max_seq_len = 128;
  Engine engine(cfg);
  const auto outs = engine.generate(make_requests(4, 40, 40, 9));
  int preemptions = 0;
  for (const auto& o : outs) {
    EXPECT_EQ(o.output_tokens.size(), static_cast<size_t>(40));
    preemptions += o.num_preemptions;
  }
  EXPECT_TRUE(preemptions > 0);
  check_greedy(path, outs, 0.1f);
}

TEST(engine_stop_tokens_and_sampling_determinism) {
  const std::string& path = model_path(QuantType::kFP16);
  Engine engine(small_config(path));
  // Stop on whatever greedy produces third.
  auto reqs = make_requests(1, 20, 10, 11);
  const auto base = engine.generate(reqs);
  reqs[0].params.stop_token_ids = {base[0].output_tokens[2]};
  const auto stopped = engine.generate(reqs);
  EXPECT_TRUE(stopped[0].finish_reason == FinishReason::kStop);
  EXPECT_TRUE(stopped[0].output_tokens.size() <= 3);

  auto sreqs = make_requests(3, 20, 20, 12);
  for (auto& r : sreqs) {
    r.params.temperature = 0.9f;
    r.params.top_p = 0.9f;
    r.params.top_k = 40;
    r.params.seed = 1000 + r.id;
  }
  const auto a = engine.generate(sreqs);
  const auto b = engine.generate(sreqs);
  for (size_t i = 0; i < a.size(); ++i) EXPECT_TRUE(a[i].output_tokens == b[i].output_tokens);
}

TEST(engine_speculative_decoding) {
  const std::string& path = model_path(QuantType::kFP16);
  {
    // Draft == target: (almost) every draft token is accepted.
    EngineConfig cfg = small_config(path);
    cfg.draft_model_path = path;
    cfg.num_speculative_tokens = 4;
    Engine engine(cfg);
    const auto outs = engine.generate(make_requests(5, 30, 33, 13));
    int64_t proposed = 0, accepted = 0;
    for (const auto& o : outs) {
      EXPECT_EQ(o.output_tokens.size(), static_cast<size_t>(33));
      proposed += o.draft_proposed;
      accepted += o.draft_accepted;
    }
    const double rate = static_cast<double>(accepted) / proposed;
    std::printf("    self-draft acceptance rate %.3f\n", rate);
    EXPECT_TRUE(rate > 0.9);
    check_greedy(path, outs, 0.1f);
  }
  {
    // Unrelated draft: low acceptance, outputs still exactly target-greedy.
    EngineConfig cfg = small_config(path);
    cfg.draft_model_path = model_path(QuantType::kINT4, 2);
    cfg.num_speculative_tokens = 3;
    Engine engine(cfg);
    const auto outs = engine.generate(make_requests(4, 25, 20, 14));
    check_greedy(path, outs, 0.1f);
  }
}

TEST(engine_benchmark_harness_report) {
  const std::string& path = model_path(QuantType::kINT8);
  EngineConfig cfg = small_config(path);
  cfg.track_memory = true;
  Engine engine(cfg);
  Workload w = synthetic_workload(8, 64, 32, engine.model_config().vocab_size, 3);
  BenchOptions opts;
  const Json r = run_benchmark(engine, w, opts);
  EXPECT_EQ(r["num_requests"].as_int(), 8);
  EXPECT_EQ(r["output_tokens"].as_int(), 8 * 32);
  EXPECT_TRUE(r["decode_throughput_tok_s"].as_double() > 0);
  EXPECT_TRUE(r["ttft_ms"]["p50"].as_double() > 0);
  EXPECT_TRUE(r["itl_ms"]["count"].as_int() == 8 * 31);
  EXPECT_TRUE(r["memory"]["peak_device_used_bytes"].as_int() > 0);
  std::printf("    %s\n", format_report_summary(r).c_str());
}

int main(int argc, char** argv) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("no CUDA device available; skipping GPU tests\n");
    return 0;
  }
  return rltest::run_all(argc, argv);
}
