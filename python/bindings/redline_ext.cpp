// pybind11 bindings for the Redline engine (module `redline._redline`).
// Results cross the boundary as JSON strings / plain lists, keeping the
// binding surface small; python/redline/runtime.py wraps it.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/workload.h"
#include "cuda/bench/harness.h"
#include "cuda/engine/engine.h"

namespace py = pybind11;
using namespace redline;

namespace {

EngineConfig config_from_dict(const py::dict& d) {
  EngineConfig c;
  // Both ternary arms must be py::object: with a bare py::none() arm the result type is py::none
  // and every present value fails the cast.
  auto get = [&](const char* k) -> py::object {
    return d.contains(k) ? py::object(d[k]) : py::object(py::none());
  };
  c.model_path = d["model"].cast<std::string>();
  if (!get("draft_model").is_none()) c.draft_model_path = get("draft_model").cast<std::string>();
  if (!get("spec_tokens").is_none()) c.num_speculative_tokens = get("spec_tokens").cast<int>();
  if (!get("max_seqs").is_none()) c.max_num_seqs = get("max_seqs").cast<int>();
  if (!get("max_batched_tokens").is_none()) c.max_num_batched_tokens = get("max_batched_tokens").cast<int>();
  if (!get("max_seq_len").is_none()) c.max_seq_len = get("max_seq_len").cast<int>();
  if (!get("max_logit_rows").is_none()) c.max_logit_rows = get("max_logit_rows").cast<int>();
  if (!get("kv_cache_tokens").is_none()) c.kv_cache_tokens = get("kv_cache_tokens").cast<int64_t>();
  if (!get("gpu_memory_utilization").is_none())
    c.gpu_memory_utilization = get("gpu_memory_utilization").cast<double>();
  if (!get("kv_dtype").is_none()) c.kv_dtype = kv_dtype_from_name(get("kv_dtype").cast<std::string>());
  if (!get("cuda_graphs").is_none()) c.cuda_graphs = get("cuda_graphs").cast<bool>();
  if (!get("async_scheduling").is_none()) c.async_scheduling = get("async_scheduling").cast<bool>();
  if (!get("mix_prefill_decode").is_none()) c.mix_prefill_decode = get("mix_prefill_decode").cast<bool>();
  if (!get("int8_gemm").is_none())
    c.linear.int8_prefill = prefill_gemm_from_name(get("int8_gemm").cast<std::string>());
  if (!get("int4_gemm").is_none())
    c.linear.int4_prefill = prefill_gemm_from_name(get("int4_gemm").cast<std::string>());
  if (!get("gemv_max_m").is_none()) c.linear.gemv_max_m = get("gemv_max_m").cast<int>();
  if (!get("int4_kernel").is_none()) c.linear.int4_tensor_cores = get("int4_kernel").cast<std::string>() == "tc";
  if (!get("track_memory").is_none()) c.track_memory = get("track_memory").cast<bool>();
  if (!get("device").is_none()) c.device = get("device").cast<int>();
  return c;
}

py::dict output_to_dict(const RequestOutput& o) {
  py::dict d;
  d["id"] = o.id;
  d["prompt_ids"] = o.prompt;
  d["output_ids"] = o.output_tokens;
  d["logprobs"] = o.logprobs;
  d["finish_reason"] = finish_reason_name(o.finish_reason);
  d["ttft_s"] = o.first_token_time - o.arrival_time;
  d["e2e_s"] = o.finish_time - o.arrival_time;
  d["token_times"] = o.token_times;
  d["draft_proposed"] = o.draft_proposed;
  d["draft_accepted"] = o.draft_accepted;
  return d;
}

}  // namespace

PYBIND11_MODULE(_redline, m) {
  m.doc() = "Redline low-latency CUDA transformer runtime";
  py::class_<Engine>(m, "Engine")
      .def(py::init([](const py::dict& cfg) { return std::make_unique<Engine>(config_from_dict(cfg)); }))
      .def(
          "generate",
          [](Engine& e, const std::vector<std::vector<int32_t>>& prompts, int max_new_tokens,
             float temperature, int top_k, float top_p, uint64_t seed, bool ignore_eos,
             const std::vector<int32_t>& stop_token_ids) {
            std::vector<Request> reqs;
            for (size_t i = 0; i < prompts.size(); ++i) {
              Request r;
              r.id = static_cast<int64_t>(i);
              r.prompt = prompts[i];
              r.params.max_new_tokens = max_new_tokens;
              r.params.temperature = temperature;
              r.params.top_k = top_k;
              r.params.top_p = top_p;
              r.params.seed = seed + i;
              r.params.ignore_eos = ignore_eos;
              r.params.stop_token_ids = stop_token_ids;
              reqs.push_back(std::move(r));
            }
            std::vector<RequestOutput> outs;
            {
              py::gil_scoped_release release;
              outs = e.generate(std::move(reqs));
            }
            py::list result;
            for (const auto& o : outs) result.append(output_to_dict(o));
            return result;
          },
          py::arg("prompts"), py::arg("max_new_tokens") = 128, py::arg("temperature") = 0.0f,
          py::arg("top_k") = 0, py::arg("top_p") = 1.0f, py::arg("seed") = 0, py::arg("ignore_eos") = false,
          py::arg("stop_token_ids") = std::vector<int32_t>{})
      .def(
          "score",
          [](Engine& e, const std::vector<int32_t>& tokens) {
            py::gil_scoped_release release;
            return e.score(tokens);
          },
          py::arg("tokens"), "log p(tokens[i] | tokens[:i]) for i >= 1")
      .def(
          "benchmark",
          [](Engine& e, const std::string& workload_json, bool online, int warmup) {
            Workload w = parse_workload(Json::parse(workload_json), e.model_config().vocab_size);
            BenchOptions opts;
            opts.online = online;
            opts.warmup_requests = warmup;
            py::gil_scoped_release release;
            return run_benchmark(e, w, opts).dump();
          },
          py::arg("workload_json"), py::arg("online") = false, py::arg("warmup") = 2)
      .def("memory_report", [](Engine& e) { return e.memory_report().dump(); })
      .def("describe", [](const Engine& e) { return e.describe().dump(); })
      .def_property_readonly("vocab_size", [](const Engine& e) { return e.model_config().vocab_size; });
}
