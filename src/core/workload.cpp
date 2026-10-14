#include "core/workload.h"

#include <algorithm>
#include <random>

#include "core/common.h"

namespace redline {
namespace {

std::vector<int32_t> random_prompt(int len, int low, int high, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> dist(low, high - 1);
  std::vector<int32_t> ids(len);
  for (auto& t : ids) t = dist(rng);
  return ids;
}

}  // namespace

int64_t Workload::total_prompt_tokens() const {
  int64_t n = 0;
  for (const auto& r : requests) n += static_cast<int64_t>(r.prompt.size());
  return n;
}

int64_t Workload::total_max_new_tokens() const {
  int64_t n = 0;
  for (const auto& r : requests) n += r.params.max_new_tokens;
  return n;
}

Workload parse_workload(const Json& doc, int vocab_size) {
  Workload w;
  w.name = doc.get_string("name", "workload");
  w.description = doc.get_string("description", "");
  if (const Json* s = doc.find("settings")) w.raw_settings = *s;
  const uint64_t seed = static_cast<uint64_t>(doc.get_int("seed", 0));
  std::mt19937_64 rng(seed);
  // Avoid the low ids, which are usually special tokens.
  const int low = static_cast<int>(doc.get_int("vocab_low", std::min(1000, vocab_size / 4)));
  const int high = static_cast<int>(doc.get_int("vocab_high", vocab_size));
  RL_CHECK(low < high && high <= vocab_size, "bad synthetic vocab range");

  const Json& reqs = doc["requests"];
  int64_t next_id = 0;
  for (const auto& r : reqs.as_array()) {
    Request req;
    req.id = r.get_int("id", next_id);
    next_id = req.id + 1;
    if (const Json* ids = r.find("prompt_ids")) {
      for (const auto& t : ids->as_array()) {
        const int64_t v = t.as_int();
        RL_CHECK(v >= 0 && v < vocab_size, "prompt token id ", v, " out of range");
        req.prompt.push_back(static_cast<int32_t>(v));
      }
    } else {
      req.prompt = random_prompt(static_cast<int>(r["prompt_len"].as_int()), low, high, rng);
    }
    RL_CHECK(!req.prompt.empty(), "request ", req.id, " has an empty prompt");
    req.params.max_new_tokens = static_cast<int>(r.get_int("max_new_tokens", 128));
    req.params.temperature = static_cast<float>(r.get_double("temperature", 0.0));
    req.params.top_k = static_cast<int>(r.get_int("top_k", 0));
    req.params.top_p = static_cast<float>(r.get_double("top_p", 1.0));
    req.params.seed = static_cast<uint64_t>(r.get_int("seed", req.id));
    req.params.ignore_eos = r.get_bool("ignore_eos", true);
    if (const Json* stops = r.find("stop_token_ids")) {
      for (const auto& t : stops->as_array()) {
        req.params.stop_token_ids.push_back(static_cast<int32_t>(t.as_int()));
      }
    }
    req.arrival_time = r.get_double("arrival_time", 0.0);
    w.requests.push_back(std::move(req));
  }
  return w;
}

Workload load_workload(const std::string& path, int vocab_size) {
  return parse_workload(Json::parse_file(path), vocab_size);
}

Workload synthetic_workload(int num_requests, int prompt_len, int output_len, int vocab_size,
                            uint64_t seed) {
  Workload w;
  w.name = str_cat("synthetic_n", num_requests, "_in", prompt_len, "_out", output_len);
  std::mt19937_64 rng(seed);
  const int low = std::min(1000, vocab_size / 4);
  for (int i = 0; i < num_requests; ++i) {
    Request r;
    r.id = i;
    r.prompt = random_prompt(prompt_len, low, vocab_size, rng);
    r.params.max_new_tokens = output_len;
    r.params.ignore_eos = true;
    r.params.seed = static_cast<uint64_t>(i);
    w.requests.push_back(std::move(r));
  }
  Json s = Json::object();
  s.set("num_requests", num_requests);
  s.set("prompt_len", prompt_len);
  s.set("output_len", output_len);
  w.raw_settings = s;
  return w;
}

}  // namespace redline
