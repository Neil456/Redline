// Shared command-line parsing for the Redline tools.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "core/common.h"
#include "cuda/engine/engine.h"

namespace redline::cli {

class Args {
 public:
  Args(int argc, char** argv);
  bool has(const std::string& key) const;
  std::string get(const std::string& key, const std::string& def = "") const;
  int get_int(const std::string& key, int def) const;
  double get_double(const std::string& key, double def) const;
  bool flag(const std::string& key) const { return has(key); }
  std::vector<std::string> get_all(const std::string& key) const;
  // Throws if any option was not consumed by a get*/has call.
  void check_unknown() const;

 private:
  std::multimap<std::string, std::string> kv_;
  mutable std::map<std::string, bool> used_;
};

// Engine options shared by all tools.
EngineConfig engine_config_from_args(const Args& args);
const char* engine_options_help();

std::vector<int32_t> parse_id_list(const std::string& s);

}  // namespace redline::cli
