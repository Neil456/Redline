#include "cuda/bench/harness.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

#include "core/common.h"
#include "core/stats.h"
#include "cuda/nvtx.h"

namespace redline {
namespace {

void warmup(Engine& engine, const Workload& w, const BenchOptions& opts) {
  RL_NVTX_RANGE("bench/warmup", kNvtxHost);
  std::vector<Request> reqs;
  for (int i = 0; i < std::min<int>(opts.warmup_requests, static_cast<int>(w.requests.size())); ++i) {
    Request r = w.requests[i];
    r.id = -1000 - i;
    r.arrival_time = 0;
    r.params.max_new_tokens = std::min(r.params.max_new_tokens, opts.warmup_tokens);
    reqs.push_back(std::move(r));
  }
  if (!reqs.empty()) engine.generate(std::move(reqs));
  engine.synchronize();
}

}  // namespace

// Compact trace: arrays of rows plus a field list, times in ms since t0.
static void write_trace(const std::string& path, const Json& report, const std::vector<StepRecord>& steps,
                 const std::vector<RequestOutput>& outputs, double t0) {
  auto ms = [&](double t) { return std::round((t - t0) * 1e6) / 1e3; };
  Json trace = Json::object();
  trace.set("format", "redline-trace-v1");
  for (const char* k : {"workload", "workload_settings", "engine", "duration_s", "output_tokens",
                        "decode_throughput_tok_s", "output_throughput_tok_s", "ttft_ms", "itl_ms",
                        "tpot_ms", "speculative", "memory"}) {
    if (const Json* v = report.find(k)) trace.set(k, *v);
  }
  Json step_fields = Json::array();
  for (const char* f : {"step", "kind", "t_launch_ms", "t_done_ms", "host_ms", "gpu_ms", "seqs",
                        "tokens", "prefill_tokens", "decode_tokens", "tokens_out", "cuda_graph",
                        "kv_used", "kv_total", "running", "waiting", "draft_proposed",
                        "draft_accepted"}) {
    step_fields.push_back(f);
  }
  Json rows = Json::array();
  for (const auto& s : steps) {
    Json r = Json::array();
    r.push_back(s.step);
    r.push_back(s.kind);
    r.push_back(ms(s.t_launch));
    r.push_back(ms(s.t_done));
    r.push_back(std::round(s.host_ms * 1000) / 1000);
    r.push_back(std::round(s.gpu_ms * 1000) / 1000);
    for (int v : {s.num_seqs, s.num_tokens, s.num_prefill_tokens, s.num_decode_tokens, s.tokens_out,
                  s.cuda_graph ? 1 : 0, s.kv_blocks_used, s.kv_blocks_total, s.num_running,
                  s.num_waiting, s.draft_proposed, s.draft_accepted}) {
      r.push_back(v);
    }
    rows.push_back(r);
  }
  trace.set("step_fields", step_fields);
  trace.set("steps", rows);
  Json req_fields = Json::array();
  for (const char* f : {"id", "arrival_ms", "first_token_ms", "finish_ms", "prompt_tokens",
                        "output_tokens", "preemptions", "draft_proposed", "draft_accepted"}) {
    req_fields.push_back(f);
  }
  Json reqs = Json::array();
  for (const auto& o : outputs) {
    Json r = Json::array();
    r.push_back(o.id);
    r.push_back(ms(o.arrival_time));
    r.push_back(ms(o.first_token_time));
    r.push_back(ms(o.finish_time));
    r.push_back(static_cast<int64_t>(o.prompt.size()));
    r.push_back(static_cast<int64_t>(o.output_tokens.size()));
    r.push_back(o.num_preemptions);
    r.push_back(o.draft_proposed);
    r.push_back(o.draft_accepted);
    reqs.push_back(r);
  }
  trace.set("request_fields", req_fields);
  trace.set("requests", reqs);
  trace.dump_file(path, -1);
}

Json run_benchmark(Engine& engine, const Workload& workload, const BenchOptions& opts) {
  warmup(engine, workload, opts);
  engine.clear_step_records();
  engine.set_record_steps(true);
  engine.reset_peak_memory();

  std::vector<const Request*> order;
  for (const auto& r : workload.requests) order.push_back(&r);
  std::stable_sort(order.begin(), order.end(), [&](const Request* a, const Request* b) {
    return (opts.online ? a->arrival_time : 0.0) < (opts.online ? b->arrival_time : 0.0);
  });

  std::vector<RequestOutput> outputs;
  size_t next = 0;
  const double t0 = now_seconds();
  {
    RL_NVTX_RANGE("bench/run", kNvtxHost);
    while (next < order.size() || engine.has_unfinished()) {
      const double now = now_seconds();
      while (next < order.size() && (!opts.online || t0 + order[next]->arrival_time <= now)) {
        Request r = *order[next];
        r.arrival_time = t0 + (opts.online ? r.arrival_time : 0.0);
        engine.add_request(std::move(r));
        ++next;
      }
      if (!engine.has_unfinished()) {
        // Idle until the next arrival.
        const double wait = t0 + order[next]->arrival_time - now_seconds();
        if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
        continue;
      }
      for (auto& o : engine.step()) outputs.push_back(std::move(o));
    }
  }
  const double t_end = now_seconds();
  engine.set_record_steps(false);
  const double duration = t_end - t0;

  // ---- per-request metrics ----
  std::vector<double> ttft, e2e, tpot, itl;
  int64_t out_tokens = 0, prompt_tokens = 0, proposed = 0, accepted = 0, preemptions = 0;
  Json per_request = Json::array();
  for (const auto& o : outputs) {
    const int n = static_cast<int>(o.output_tokens.size());
    out_tokens += n;
    prompt_tokens += static_cast<int64_t>(o.prompt.size());
    proposed += o.draft_proposed;
    accepted += o.draft_accepted;
    preemptions += o.num_preemptions;
    const double t_first = (o.first_token_time - o.arrival_time) * 1000.0;
    const double t_e2e = (o.finish_time - o.arrival_time) * 1000.0;
    ttft.push_back(t_first);
    e2e.push_back(t_e2e);
    if (n > 1) tpot.push_back((o.finish_time - o.first_token_time) * 1000.0 / (n - 1));
    for (size_t i = 1; i < o.token_times.size(); ++i) {
      itl.push_back((o.token_times[i] - o.token_times[i - 1]) * 1000.0);
    }
    if (opts.keep_per_request) {
      Json r = Json::object();
      r.set("id", o.id);
      r.set("prompt_tokens", static_cast<int64_t>(o.prompt.size()));
      r.set("output_tokens", n);
      r.set("ttft_ms", t_first);
      r.set("e2e_ms", t_e2e);
      r.set("finish_reason", finish_reason_name(o.finish_reason));
      per_request.push_back(r);
    }
  }

  // ---- per-step metrics (CUDA events + host timestamps) ----
  const auto& steps = engine.step_records();
  double decode_gpu_ms = 0, decode_wall_s = 0, prefill_gpu_ms = 0;
  int64_t decode_tokens = 0, prefill_tokens = 0, graph_steps = 0, spec_seq_steps = 0, spec_tokens = 0;
  std::vector<double> decode_step_gpu, host_overhead;
  double prev_done = t0;
  for (const auto& s : steps) {
    const bool decode_like = s.kind == "decode" || s.kind == "spec";
    host_overhead.push_back(s.host_ms);
    if (s.cuda_graph) ++graph_steps;
    if (decode_like) {
      decode_gpu_ms += s.gpu_ms;
      decode_tokens += s.tokens_out;
      decode_wall_s += s.t_done - prev_done;
      decode_step_gpu.push_back(s.gpu_ms);
    } else {
      prefill_gpu_ms += s.gpu_ms;
      prefill_tokens += s.num_prefill_tokens;
    }
    if (s.kind == "spec") {
      spec_seq_steps += s.num_seqs;
      spec_tokens += s.tokens_out;
    }
    prev_done = s.t_done;
  }

  Json report = Json::object();
  report.set("workload", workload.name);
  report.set("workload_settings", workload.raw_settings);
  report.set("mode", opts.online ? "online" : "offline");
  report.set("engine", engine.describe());
  report.set("num_requests", static_cast<int64_t>(outputs.size()));
  report.set("duration_s", duration);
  report.set("prompt_tokens", prompt_tokens);
  report.set("output_tokens", out_tokens);
  report.set("request_throughput_rps", outputs.size() / duration);
  report.set("output_throughput_tok_s", out_tokens / duration);
  report.set("decode_throughput_tok_s", decode_wall_s > 0 ? decode_tokens / decode_wall_s : 0.0);
  report.set("decode_throughput_gpu_tok_s",
             decode_gpu_ms > 0 ? decode_tokens / (decode_gpu_ms / 1000.0) : 0.0);
  report.set("prefill_throughput_gpu_tok_s",
             prefill_gpu_ms > 0 ? prefill_tokens / (prefill_gpu_ms / 1000.0) : 0.0);
  report.set("ttft_ms", summarize(ttft).to_json());
  report.set("itl_ms", summarize(itl).to_json());
  report.set("tpot_ms", summarize(tpot).to_json());
  report.set("e2e_latency_ms", summarize(e2e).to_json());
  report.set("decode_step_gpu_ms", summarize(decode_step_gpu).to_json());
  report.set("host_overhead_ms", summarize(host_overhead).to_json());
  report.set("num_steps", static_cast<int64_t>(steps.size()));
  report.set("cuda_graph_step_fraction", steps.empty() ? 0.0 : static_cast<double>(graph_steps) / steps.size());
  report.set("preemptions", preemptions);
  Json spec = Json::object();
  spec.set("enabled", engine.speculative());
  spec.set("draft_tokens_proposed", proposed);
  spec.set("draft_tokens_accepted", accepted);
  spec.set("acceptance_rate", proposed > 0 ? static_cast<double>(accepted) / proposed : 0.0);
  spec.set("tokens_per_seq_step", spec_seq_steps > 0 ? static_cast<double>(spec_tokens) / spec_seq_steps : 0.0);
  report.set("speculative", spec);
  report.set("memory", engine.memory_report());
  if (opts.keep_per_request) report.set("requests", per_request);
  if (!opts.trace_path.empty()) write_trace(opts.trace_path, report, steps, outputs, t0);
  return report;
}

std::string format_report_summary(const Json& r) {
  char buf[512];
  const double peak = static_cast<double>(r["memory"]["peak_device_used_bytes"].as_int()) / (1 << 30);
  std::snprintf(buf, sizeof(buf),
                "%s: %lld req, out %.1f tok/s, decode %.1f tok/s, TTFT p50 %.2f ms, ITL p50 %.2f ms "
                "(p99 %.2f), peak VRAM %.2f GiB%s",
                r["workload"].as_string().c_str(), static_cast<long long>(r["num_requests"].as_int()),
                r["output_throughput_tok_s"].as_double(), r["decode_throughput_tok_s"].as_double(),
                r["ttft_ms"]["p50"].as_double(), r["itl_ms"]["p50"].as_double(),
                r["itl_ms"]["p99"].as_double(), peak,
                r["speculative"]["enabled"].as_bool()
                    ? str_cat(", acceptance ", r["speculative"]["acceptance_rate"].as_double()).c_str()
                    : "");
  return buf;
}

}  // namespace redline
