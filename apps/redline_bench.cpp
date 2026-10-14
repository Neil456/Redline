// Benchmark driver. Runs one or more workloads (JSON files produced by
// `python -m redline.workloads`, or fixed-shape synthetic ones) and writes a
// JSON report per workload. Profile with Nsight Systems to see the NVTX ranges:
//
//   nsys profile -t cuda,nvtx -o redline redline_bench --model m.rlm --synthetic 8,512,256
//
//   redline_bench --model m.rlm --workload w1.json --workload w2.json --out-dir results/
#include <cstdio>
#include <sys/stat.h>

#include "cli_common.h"
#include "core/json.h"
#include "core/workload.h"
#include "cuda/bench/harness.h"

using namespace redline;

namespace {

std::string basename_no_ext(const std::string& path) {
  std::string b = path.substr(path.find_last_of('/') + 1);
  const auto dot = b.find_last_of('.');
  return dot == std::string::npos ? b : b.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    cli::Args args(argc, argv);
    if (args.flag("help")) {
      std::printf(
          "usage: redline_bench --model PATH (--workload FILE ... | --synthetic N,IN,OUT ...)\n"
          "  --out FILE | --out-dir DIR   report destination (default: stdout summary only)\n"
          "  --online                     honor workload arrival times\n"
          "  --warmup N                   warm-up requests (default 2)\n"
          "  --per-request                include per-request metrics\n"
          "  --trace-out FILE             step-by-step trace for demo/index.html\n"
          "  --tag NAME                   label stored in the report\n%s",
          cli::engine_options_help());
      return 0;
    }
    EngineConfig cfg = cli::engine_config_from_args(args);
    BenchOptions opts;
    opts.online = args.flag("online");
    opts.warmup_requests = args.get_int("warmup", opts.warmup_requests);
    opts.keep_per_request = args.flag("per-request");
    const std::string out_file = args.get("out");
    const std::string out_dir = args.get("out-dir");
    const std::string tag = args.get("tag", "redline");
    const std::string trace_out = args.get("trace-out");
    const auto workload_files = args.get_all("workload");
    const auto synthetic = args.get_all("synthetic");
    args.check_unknown();
    if (workload_files.empty() && synthetic.empty()) throw Error("give --workload or --synthetic");

    Engine engine(cfg);
    std::vector<std::pair<std::string, Workload>> workloads;
    for (const auto& f : workload_files) {
      workloads.emplace_back(basename_no_ext(f), load_workload(f, engine.model_config().vocab_size));
    }
    for (const auto& s : synthetic) {
      const auto v = cli::parse_id_list(s);
      if (v.size() != 3) throw Error("--synthetic expects N,IN,OUT");
      Workload w = synthetic_workload(v[0], v[1], v[2], engine.model_config().vocab_size, 0);
      workloads.emplace_back(w.name, std::move(w));
    }
    if (!out_dir.empty()) ::mkdir(out_dir.c_str(), 0755);
    Json all = Json::array();
    for (auto& [name, w] : workloads) {
      if (!trace_out.empty()) {
        // One trace per workload; add the workload name when there are several.
        std::string path = trace_out;
        if (workloads.size() > 1) {
          const auto dot = path.rfind(".json");
          path = (dot == std::string::npos ? path : path.substr(0, dot)) + "__" + name + ".json";
        }
        opts.trace_path = path;
      }
      Json report = run_benchmark(engine, w, opts);
      report.set("tag", tag);
      report.set("system", "redline");
      std::fprintf(stderr, "%s\n", format_report_summary(report).c_str());
      if (!out_dir.empty()) report.dump_file(out_dir + "/" + tag + "__" + name + ".json");
      all.push_back(std::move(report));
    }
    if (!out_file.empty()) {
      (all.size() == 1 ? all[0] : all).dump_file(out_file);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
