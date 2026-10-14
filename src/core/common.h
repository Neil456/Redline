// Shared host-side utilities: error checking, logging, small math helpers.
#pragma once

#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace redline {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

template <typename... Args>
std::string str_cat(Args&&... args) {
  std::ostringstream os;
  (os << ... << std::forward<Args>(args));
  return os.str();
}

#define RL_CHECK(cond, ...)                                                        \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      throw ::redline::Error(::redline::str_cat(__FILE__, ":", __LINE__,           \
                                                ": check failed: " #cond " ",      \
                                                ##__VA_ARGS__));                   \
    }                                                                              \
  } while (0)

enum class LogLevel { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

void set_log_level(LogLevel level);
LogLevel log_level();
void log_message(LogLevel level, const std::string& msg);

#define RL_LOG(level, ...)                                                   \
  do {                                                                       \
    if (static_cast<int>(level) >= static_cast<int>(::redline::log_level())) \
      ::redline::log_message(level, ::redline::str_cat(__VA_ARGS__));        \
  } while (0)
#define RL_DEBUG(...) RL_LOG(::redline::LogLevel::kDebug, __VA_ARGS__)
#define RL_INFO(...) RL_LOG(::redline::LogLevel::kInfo, __VA_ARGS__)
#define RL_WARN(...) RL_LOG(::redline::LogLevel::kWarn, __VA_ARGS__)

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int64_t round_up(int64_t a, int64_t b) { return ceil_div(a, b) * b; }

// Monotonic wall clock in seconds.
inline double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// IEEE fp16 <-> fp32 conversion on the host (round-to-nearest-even).
float half_bits_to_float(uint16_t h);
uint16_t float_to_half_bits(float f);

std::string format_bytes(uint64_t bytes);

}  // namespace redline
