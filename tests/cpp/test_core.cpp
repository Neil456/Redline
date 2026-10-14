// Host-only unit tests: JSON, safetensors, fp16 conversion, config/RoPE,
// block allocator, scheduler, statistics, workloads.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "core/batch_builder.h"
#include "core/block_allocator.h"
#include "core/common.h"
#include "core/json.h"
#include "core/model_config.h"
#include "core/safetensors.h"
#include "core/scheduler.h"
#include "core/stats.h"
#include "core/workload.h"
#include "test_framework.h"

using namespace redline;

namespace {

std::string tmp_path(const std::string& name) {
  const char* dir = std::getenv("TMPDIR");
  return std::string(dir ? dir : "/tmp") + "/redline_test_" + name;
}

std::unique_ptr<Sequence> make_seq(int64_t id, int prompt_len, int max_new) {
  auto s = std::make_unique<Sequence>();
  s->id = id;
  for (int i = 0; i < prompt_len; ++i) s->tokens.push_back(100 + i);
  s->prompt_len = prompt_len;
  s->params.max_new_tokens = max_new;
  return s;
}

// Simulates the engine receiving one sampled token for each sampling entry.
void deliver(Scheduler& sched, const StepPlan& plan, int32_t token) {
  for (const auto& e : plan.seqs) {
    if (!e.sample) continue;
    Sequence* s = e.seq;
    s->tokens.push_back(token);
    s->num_pending--;
    if (s->status != SeqStatus::kFinished && s->num_pending == 0 &&
        s->num_generated() >= s->params.max_new_tokens) {
      sched.finish(s, FinishReason::kLength);
    }
    sched.on_tokens_received(s);
  }
}

// Marks sampled entries as pending (what the engine does at enqueue time).
void mark_pending(StepPlan& plan) {
  int row = 0;
  for (auto& e : plan.seqs) {
    if (!e.sample) continue;
    e.seq->num_pending++;
    e.seq->pending_step = plan.step_id;
    e.seq->pending_row = row++;
  }
}

}  // namespace

TEST(json_roundtrip) {
  const std::string text =
      R"({"a": 1, "b": [true, false, null, 2.5, -3e2], "c": {"s": "x\"yé\n"}, "d": []})";
  Json j = Json::parse(text);
  EXPECT_EQ(j["a"].as_int(), 1);
  EXPECT_EQ(j["b"].size(), static_cast<size_t>(5));
  EXPECT_TRUE(j["b"][0].as_bool());
  EXPECT_TRUE(j["b"][2].is_null());
  EXPECT_NEAR(j["b"][3].as_double(), 2.5, 0);
  EXPECT_NEAR(j["b"][4].as_double(), -300.0, 0);
  EXPECT_EQ(j["c"]["s"].as_string(), std::string("x\"y\xc3\xa9\n"));
  Json again = Json::parse(j.dump(2));
  EXPECT_EQ(again.dump(), j.dump());
  EXPECT_EQ(j.get_int("missing", 7), 7);
  EXPECT_THROWS(Json::parse("{\"a\": }"));
  EXPECT_THROWS(Json::parse("[1, 2"));
}

TEST(json_numbers) {
  Json j = Json::object();
  j.set("x", 0.1);
  j.set("big", static_cast<int64_t>(1) << 40);
  Json back = Json::parse(j.dump());
  EXPECT_NEAR(back["x"].as_double(), 0.1, 0);
  EXPECT_EQ(back["big"].as_int(), static_cast<int64_t>(1) << 40);
}

TEST(half_conversion) {
  const float values[] = {0.0f, 1.0f, -2.5f, 65504.0f, 6.1035156e-05f, 5.9604645e-08f, 0.1f, 1e-3f};
  for (float v : values) {
    const float back = half_bits_to_float(float_to_half_bits(v));
    EXPECT_NEAR(back, v, std::fabs(v) * 1e-3 + 1e-8);
  }
  EXPECT_EQ(float_to_half_bits(1.0f), 0x3C00);
  EXPECT_EQ(float_to_half_bits(-2.0f), 0xC000);
  EXPECT_EQ(float_to_half_bits(1e6f), 0x7C00);   // overflow -> inf
  EXPECT_EQ(float_to_half_bits(1.0009765625f), 0x3C01);
  // Round-to-nearest-even at the midpoint between 1 and 1+2^-10.
  EXPECT_EQ(float_to_half_bits(1.00048828125f), 0x3C00);
  for (uint32_t h = 0; h < 0x7C00; h += 7) {  // all finite positive halves round-trip
    EXPECT_EQ(float_to_half_bits(half_bits_to_float(static_cast<uint16_t>(h))), h);
  }
}

TEST(safetensors_roundtrip) {
  SafetensorsWriter w;
  std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<int8_t> b = {-1, 2, -3};
  w.add("a", DType::kF32, {2, 3}, a.data(), a.size() * 4);
  w.add("b", DType::kI8, {3}, b.data(), b.size());
  w.set_metadata("k", "v");
  const std::string path = tmp_path("st.safetensors");
  w.write(path);
  SafetensorsFile f(path);
  EXPECT_TRUE(f.has("a") && f.has("b") && !f.has("c"));
  EXPECT_EQ(f.info("a").shape_str(), std::string("[2, 3]"));
  EXPECT_EQ(f.metadata().at("k"), std::string("v"));
  const float* pa = static_cast<const float*>(f.data("a"));
  EXPECT_NEAR(pa[5], 6.0, 0);
  const int8_t* pb = static_cast<const int8_t*>(f.data("b"));
  EXPECT_EQ(static_cast<int>(pb[2]), -3);
  std::remove(path.c_str());
}

TEST(model_config_roundtrip_and_rope) {
  ModelConfig c;
  c.vocab_size = 1000;
  c.hidden_size = 256;
  c.intermediate_size = 512;
  c.num_layers = 2;
  c.num_heads = 8;
  c.num_kv_heads = 2;
  c.head_dim = 32;
  c.rope_theta = 500000.0;
  c.rope_scaling.type = "llama3";
  c.rope_scaling.factor = 8.0;
  c.rope_scaling.low_freq_factor = 1.0;
  c.rope_scaling.high_freq_factor = 4.0;
  c.rope_scaling.original_max_position_embeddings = 8192;
  c.weight_quant = QuantType::kINT4;
  c.group_size = 64;
  c.eos_token_ids = {1, 2};
  ModelConfig d = ModelConfig::from_json(Json::parse(c.to_json().dump()));
  EXPECT_EQ(d.num_kv_heads, 2);
  EXPECT_EQ(d.gqa_group(), 4);
  EXPECT_EQ(d.qkv_size(), 8 * 32 + 2 * 2 * 32);
  EXPECT_TRUE(d.weight_quant == QuantType::kINT4);
  EXPECT_EQ(d.eos_token_ids.size(), static_cast<size_t>(2));

  // llama3 scaling: highest frequency untouched, lowest divided by factor.
  ModelConfig plain = d;
  plain.rope_scaling.type = "none";
  auto inv_plain = rope_inv_freq(plain);
  auto inv_scaled = rope_inv_freq(d);
  EXPECT_NEAR(inv_scaled[0], inv_plain[0], 1e-12);
  EXPECT_NEAR(inv_scaled.back(), inv_plain.back() / 8.0, 1e-15);

  std::vector<float> cs, sn;
  rope_cos_sin_tables(plain, 16, &cs, &sn);
  EXPECT_NEAR(cs[0], 1.0, 0);
  EXPECT_NEAR(sn[3 * 16 + 0], std::sin(3.0), 1e-6);
}

TEST(block_allocator_basic) {
  BlockAllocator a(8);
  EXPECT_EQ(a.num_free(), 7);
  std::vector<int32_t> got;
  for (int i = 0; i < 7; ++i) got.push_back(a.allocate());
  EXPECT_EQ(a.allocate(), -1);
  std::set<int32_t> uniq(got.begin(), got.end());
  EXPECT_EQ(uniq.size(), static_cast<size_t>(7));
  EXPECT_TRUE(uniq.count(0) == 0);  // scratch block never handed out
  a.free(got);
  EXPECT_EQ(a.num_free(), 7);
  EXPECT_THROWS(a.free(got[0]));  // double free
  EXPECT_THROWS(a.free(0));       // reserved
}

TEST(scheduler_prefill_then_decode) {
  BlockAllocator alloc(64);
  SchedulerConfig cfg;
  cfg.max_num_seqs = 4;
  cfg.max_num_batched_tokens = 32;
  cfg.block_size = 4;
  cfg.max_seq_len = 128;
  Scheduler sched(cfg, &alloc);
  auto a = make_seq(1, 20, 3);
  auto b = make_seq(2, 20, 3);
  sched.add(a.get());
  sched.add(b.get());

  // Step 0: a's full prompt (20) + 12 tokens of b (chunked prefill).
  StepPlan p0 = sched.schedule(0, -1);
  EXPECT_EQ(p0.seqs.size(), static_cast<size_t>(2));
  EXPECT_EQ(p0.seqs[0].num_tokens, 20);
  EXPECT_TRUE(p0.seqs[0].sample);
  EXPECT_EQ(p0.seqs[1].num_tokens, 12);
  EXPECT_TRUE(!p0.seqs[1].sample);
  EXPECT_EQ(static_cast<int>(a->blocks.size()), 5);
  mark_pending(p0);

  // Step 1 is planned before step 0's token reaches the host (async).
  StepPlan p1 = sched.schedule(1, 0);
  EXPECT_EQ(p1.seqs.size(), static_cast<size_t>(2));
  EXPECT_EQ(p1.seqs[0].seq, a.get());
  EXPECT_TRUE(p1.seqs[0].input_from_device);
  EXPECT_EQ(p1.seqs[0].src_row, 0);
  EXPECT_EQ(p1.seqs[0].start_pos, 20);
  EXPECT_EQ(p1.seqs[1].num_tokens, 8);  // rest of b's prompt
  EXPECT_TRUE(p1.seqs[1].sample);
  mark_pending(p1);
  deliver(sched, p0, 7);
  EXPECT_EQ(a->num_pending, 1);
  EXPECT_EQ(static_cast<int>(a->tokens.size()), 21);

  StepPlan p2 = sched.schedule(2, 1);
  EXPECT_TRUE(p2.pure_decode());
  EXPECT_EQ(p2.seqs.size(), static_cast<size_t>(2));
  mark_pending(p2);
  deliver(sched, p1, 8);
  // a has generated 3 tokens (2 received + 1 pending) -> no more compute.
  StepPlan p3 = sched.schedule(3, 2);
  EXPECT_EQ(p3.seqs.size(), static_cast<size_t>(1));
  EXPECT_EQ(p3.seqs[0].seq, b.get());
  mark_pending(p3);
  deliver(sched, p2, 9);
  EXPECT_TRUE(a->status == SeqStatus::kFinished);
  EXPECT_EQ(a->num_generated(), 3);
  deliver(sched, p3, 10);
  EXPECT_TRUE(b->status == SeqStatus::kFinished);
  EXPECT_TRUE(!sched.has_unfinished());
  EXPECT_EQ(alloc.num_used(), 0);
}

TEST(scheduler_preempts_when_pool_exhausted) {
  // 6 usable blocks of 4 tokens; two sequences of 8 prompt tokens each take
  // 2 blocks; decoding past 12 tokens forces a preemption.
  BlockAllocator alloc(7);
  SchedulerConfig cfg;
  cfg.max_num_seqs = 4;
  cfg.max_num_batched_tokens = 64;
  cfg.block_size = 4;
  cfg.max_seq_len = 64;
  cfg.admission_watermark = 0.0;
  Scheduler sched(cfg, &alloc);
  auto a = make_seq(1, 8, 12);
  auto b = make_seq(2, 8, 12);
  sched.add(a.get());
  sched.add(b.get());
  int64_t step = 0;
  int preemptions = 0;
  while (sched.has_unfinished() && step < 200) {
    StepPlan p = sched.schedule(step, -1);  // synchronous: deliver immediately
    preemptions += p.num_preempted;
    mark_pending(p);
    deliver(sched, p, 5);
    ++step;
  }
  EXPECT_TRUE(!sched.has_unfinished());
  EXPECT_TRUE(preemptions > 0);
  EXPECT_EQ(a->num_generated(), 12);
  EXPECT_EQ(b->num_generated(), 12);
  EXPECT_EQ(alloc.num_used(), 0);
}

TEST(scheduler_lookahead_and_no_mix) {
  BlockAllocator alloc(64);
  SchedulerConfig cfg;
  cfg.block_size = 4;
  cfg.lookahead_slots = 4;
  cfg.mix_prefill_decode = false;
  cfg.max_seq_len = 128;
  Scheduler sched(cfg, &alloc);
  auto a = make_seq(1, 6, 8);
  sched.add(a.get());
  StepPlan p0 = sched.schedule(0, -1);
  mark_pending(p0);
  deliver(sched, p0, 3);
  auto b = make_seq(2, 5, 8);
  sched.add(b.get());
  // Prefill has priority when mixing is disabled.
  StepPlan p1 = sched.schedule(1, -1);
  EXPECT_EQ(p1.seqs.size(), static_cast<size_t>(1));
  EXPECT_EQ(p1.seqs[0].seq, b.get());
  mark_pending(p1);
  deliver(sched, p1, 3);
  StepPlan p2 = sched.schedule(2, -1);
  EXPECT_TRUE(p2.pure_decode());
  EXPECT_EQ(p2.seqs.size(), static_cast<size_t>(2));
  // Position 6 plus 4 lookahead slots -> 11 positions -> 3 blocks.
  EXPECT_EQ(static_cast<int>(a->blocks.size()), 3);
}

TEST(stats_percentiles) {
  std::vector<double> v;
  for (int i = 1; i <= 100; ++i) v.push_back(i);
  Summary s = summarize(v);
  EXPECT_NEAR(s.mean, 50.5, 1e-9);
  EXPECT_NEAR(s.p50, 50.5, 1e-9);
  EXPECT_NEAR(s.p99, 99.01, 1e-9);
  EXPECT_NEAR(s.min, 1, 0);
  EXPECT_NEAR(s.max, 100, 0);
  EXPECT_EQ(summarize({}).count, static_cast<size_t>(0));
}

TEST(workload_parse) {
  Json doc = Json::parse(R"({
    "name": "w", "seed": 3,
    "requests": [
      {"prompt_ids": [5, 6, 7], "max_new_tokens": 4, "temperature": 0.7, "top_p": 0.9},
      {"prompt_len": 10, "max_new_tokens": 2, "arrival_time": 0.5}
    ]})");
  Workload w = parse_workload(doc, 32000);
  EXPECT_EQ(w.requests.size(), static_cast<size_t>(2));
  EXPECT_EQ(w.requests[0].prompt.size(), static_cast<size_t>(3));
  EXPECT_NEAR(w.requests[0].params.temperature, 0.7, 1e-6);
  EXPECT_EQ(w.requests[1].prompt.size(), static_cast<size_t>(10));
  EXPECT_EQ(w.requests[1].id, 1);
  EXPECT_NEAR(w.requests[1].arrival_time, 0.5, 0);
  Workload again = parse_workload(doc, 32000);
  EXPECT_TRUE(again.requests[1].prompt == w.requests[1].prompt);  // seeded
  EXPECT_THROWS(parse_workload(Json::parse(R"({"requests": [{"prompt_ids": [40000]}]})"), 32000));
}

TEST(batch_builder_layout_and_work_lists) {
  // block_size 4, decode threshold 2: seq A decodes 1 token, seq B is a
  // 6-token prefill chunk starting at position 3, seq C verifies 2 tokens.
  BatchBuilder bb(64, 8, 16, 4, 4, 2);
  SamplingParams greedy, hot;
  hot.temperature = 0.5f;
  hot.top_k = 7;
  hot.seed = (static_cast<uint64_t>(3) << 32) | 9;
  const int32_t a_src = 5;
  EXPECT_EQ(bb.add(nullptr, 9, 1, &a_src, {11, 12, 13}, greedy, LogitsMode::kLast), 0);
  const std::vector<int32_t> b_tok = {40, 41, 42, 43, 44, 45};
  EXPECT_EQ(bb.add(b_tok.data(), 3, 6, nullptr, {20, 21, 22}, hot, LogitsMode::kNone), -1);
  const std::vector<int32_t> c_tok = {7, 0};
  const std::vector<int32_t> c_src = {-1, 2};
  EXPECT_EQ(bb.add(c_tok.data(), 0, 2, c_src.data(), {30}, greedy, LogitsMode::kAll), 1);
  EXPECT_THROWS(bb.add(b_tok.data(), 10, 6, nullptr, {1, 2}, greedy, LogitsMode::kNone));  // too few blocks

  std::vector<int32_t> buf(bb.max_meta_words(), -7);
  const BatchShape s = bb.write(buf.data());
  EXPECT_EQ(s.num_tokens, 9);
  EXPECT_EQ(s.num_seqs, 3);
  EXPECT_EQ(s.num_logit_rows, 3);
  EXPECT_EQ(s.num_decode_items, 3);  // A (1) + C (2)
  EXPECT_EQ(s.num_prefill_seqs, 1);
  EXPECT_EQ(s.max_prefill_q_tiles, 1);
  EXPECT_EQ(s.max_context, 10);
  const MetaLayout m = MetaLayout::make(s, 4);
  EXPECT_EQ(buf[m.input_src + 0], 5);
  EXPECT_EQ(buf[m.input_ids + 1], 40);
  EXPECT_EQ(buf[m.input_src + 8], 2);
  EXPECT_EQ(buf[m.positions + 0], 9);
  EXPECT_EQ(buf[m.slot_mapping + 0], 13 * 4 + 1);  // position 9 -> block 2, offset 1
  EXPECT_EQ(buf[m.slot_mapping + 1], 20 * 4 + 3);  // position 3 -> block 0, offset 3
  EXPECT_EQ(buf[m.slot_mapping + 6], 22 * 4 + 0);  // position 8
  EXPECT_EQ(buf[m.seq_q_start + 2], 7);
  EXPECT_EQ(buf[m.seq_ctx_len + 1], 9);
  EXPECT_EQ(buf[m.block_tables + 4 + 2], 22);
  EXPECT_EQ(buf[m.block_tables + 8 + 1], 0);       // padded
  EXPECT_EQ(buf[m.decode_token + 1], 7);
  EXPECT_EQ(buf[m.decode_seq + 2], 2);
  EXPECT_EQ(buf[m.prefill_seq + 0], 1);
  EXPECT_EQ(buf[m.logit_token + 0], 0);
  EXPECT_EQ(buf[m.logit_token + 2], 8);
  EXPECT_EQ(buf[m.param_row + 2], 2);
  EXPECT_EQ(buf[m.top_k + 1], 7);
  EXPECT_EQ(static_cast<uint32_t>(buf[m.seed_lo + 1]), 9u);
  EXPECT_EQ(static_cast<uint32_t>(buf[m.seed_hi + 1]), 3u);
  float t;
  std::memcpy(&t, &buf[m.temperature + 1], 4);
  EXPECT_NEAR(t, 0.5, 0);
  EXPECT_EQ(m.total_words, MetaLayout::make(bb.shape(), 4).total_words);
  EXPECT_TRUE(m.total_words <= bb.max_meta_words());
}

TEST(batch_builder_graph_padding_is_shape_stable) {
  BatchBuilder bb(64, 8, 16, 4, 16, 1);
  const int32_t tok = 5;
  bb.add(&tok, 20, 1, nullptr, {3, 4}, SamplingParams(), LogitsMode::kLast);
  bb.pad_decode_to(4);
  std::vector<int32_t> buf(bb.max_meta_words());
  const BatchShape s = bb.write(buf.data());
  BatchBuilder dummy(64, 8, 16, 4, 16, 1);
  dummy.pad_decode_to(4);
  const BatchShape d = dummy.shape();
  // Same layout as the all-dummy batch used at graph capture time.
  EXPECT_EQ(s.num_tokens, d.num_tokens);
  EXPECT_EQ(s.num_decode_items, d.num_decode_items);
  EXPECT_EQ(MetaLayout::make(s, 4).total_words, MetaLayout::make(d, 4).total_words);
  const MetaLayout m = MetaLayout::make(s, 4);
  EXPECT_EQ(buf[m.slot_mapping + 3], 0);  // padding writes the scratch block
}

RL_TEST_MAIN()
