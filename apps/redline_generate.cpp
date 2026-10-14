// Generates continuations for token-id prompts and prints one JSON object
// per request. Text in/out is handled by the Python wrapper
// (`python -m redline.generate`), which tokenizes with the HF tokenizer.
//
//   redline_generate --model m.rlm --prompt-ids 1,15043,29892 --max-new-tokens 64
//   redline_generate --model m.rlm --prompts-json prompts.json --temperature 0.8 --top-p 0.95
//   redline_generate --model m.rlm --score-ids 1,15043,29892,3186
#include <cstdio>
#include <string>
#include <vector>

#include "cli_common.h"
#include "core/json.h"

using namespace redline;

int main(int argc, char** argv) {
  try {
    cli::Args args(argc, argv);
    if (args.flag("help")) {
      std::printf(
          "usage: redline_generate --model PATH (--prompt-ids IDS ... | --prompts-json FILE | "
          "--score-ids IDS)\n"
          "  --max-new-tokens N  --temperature T  --top-k K  --top-p P  --seed S  --ignore-eos\n%s",
          cli::engine_options_help());
      return 0;
    }
    EngineConfig cfg = cli::engine_config_from_args(args);
    SamplingParams sp;
    sp.max_new_tokens = args.get_int("max-new-tokens", 128);
    sp.temperature = static_cast<float>(args.get_double("temperature", 0.0));
    sp.top_k = args.get_int("top-k", 0);
    sp.top_p = static_cast<float>(args.get_double("top-p", 1.0));
    sp.seed = static_cast<uint64_t>(args.get_int("seed", 0));
    sp.ignore_eos = args.flag("ignore-eos");

    std::vector<std::vector<int32_t>> prompts;
    for (const auto& s : args.get_all("prompt-ids")) prompts.push_back(cli::parse_id_list(s));
    const std::string prompts_json = args.get("prompts-json");
    if (!prompts_json.empty()) {
      const Json doc = Json::parse_file(prompts_json);
      for (const auto& p : doc.as_array()) {
        std::vector<int32_t> ids;
        for (const auto& t : p.as_array()) ids.push_back(static_cast<int32_t>(t.as_int()));
        prompts.push_back(std::move(ids));
      }
    }
    const std::string score_ids = args.get("score-ids");
    args.check_unknown();
    if (cfg.kv_cache_tokens == 0 && cfg.gpu_memory_utilization > 0.5) {
      cfg.gpu_memory_utilization = 0.5;  // a CLI run does not need the whole GPU
    }

    Engine engine(cfg);
    if (!score_ids.empty()) {
      const auto lp = engine.score(cli::parse_id_list(score_ids));
      Json out = Json::object();
      out.set("logprobs", Json::array_of(lp));
      std::printf("%s\n", out.dump().c_str());
      return 0;
    }
    if (prompts.empty()) throw Error("no prompts given (--prompt-ids or --prompts-json)");
    std::vector<Request> reqs;
    for (size_t i = 0; i < prompts.size(); ++i) {
      Request r;
      r.id = static_cast<int64_t>(i);
      r.prompt = prompts[i];
      r.params = sp;
      r.params.seed = sp.seed + i;
      reqs.push_back(std::move(r));
    }
    for (const auto& o : engine.generate(std::move(reqs))) {
      Json j = Json::object();
      j.set("id", o.id);
      j.set("output_ids", Json::array_of(o.output_tokens));
      j.set("logprobs", Json::array_of(o.logprobs));
      j.set("finish_reason", finish_reason_name(o.finish_reason));
      j.set("ttft_ms", (o.first_token_time - o.arrival_time) * 1000.0);
      j.set("e2e_ms", (o.finish_time - o.arrival_time) * 1000.0);
      if (o.draft_proposed > 0) {
        j.set("acceptance_rate", static_cast<double>(o.draft_accepted) / o.draft_proposed);
      }
      std::printf("%s\n", j.dump().c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
