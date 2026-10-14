// CPU reference runner: prints logits or greedy continuations computed by
// the fp32 reference implementation. Used by the Python exporter tests.
//
//   redline_ref --model m.rlm --tokens 1,2,3 --logits-out logits.f32
//   redline_ref --model m.rlm --tokens 1,2,3 --generate 16
//   redline_ref --write-random-model tiny.rlm --quant int4
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/common.h"
#include "core/json.h"
#include "cpu/random_model.h"
#include "cpu/reference_model.h"

using namespace redline;

namespace {

std::vector<int32_t> parse_ids(const std::string& s) {
  std::vector<int32_t> ids;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) ids.push_back(static_cast<int32_t>(std::stol(tok)));
  }
  return ids;
}

void usage() {
  std::fprintf(stderr,
               "usage: redline_ref --model PATH --tokens IDS [--logits-out FILE] [--generate N]\n"
               "       redline_ref --write-random-model PATH [--quant fp16|int8|int4] [--seed S]\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, tokens, logits_out, random_out, quant = "fp16";
  int generate = 0;
  uint64_t seed = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--tokens") tokens = next();
    else if (a == "--logits-out") logits_out = next();
    else if (a == "--generate") generate = std::stoi(next());
    else if (a == "--write-random-model") random_out = next();
    else if (a == "--quant") quant = next();
    else if (a == "--seed") seed = std::stoull(next());
    else {
      usage();
      return 2;
    }
  }
  try {
    if (!random_out.empty()) {
      write_random_model(random_out, tiny_test_config(quant_type_from_name(quant)), seed);
      return 0;
    }
    if (model.empty() || tokens.empty()) {
      usage();
      return 2;
    }
    CpuReferenceModel ref(model);
    const std::vector<int32_t> ids = parse_ids(tokens);
    if (!logits_out.empty()) {
      const std::vector<float> logits = ref.forward(ids);
      std::ofstream out(logits_out, std::ios::binary);
      out.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * sizeof(float)));
    }
    if (generate > 0) {
      std::printf("%s\n", Json::array_of(ref.generate_greedy(ids, generate)).dump().c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
