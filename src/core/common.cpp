#include "core/common.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace redline {
namespace {

LogLevel initial_log_level() {
  const char* env = std::getenv("REDLINE_LOG_LEVEL");
  if (env == nullptr) return LogLevel::kInfo;
  std::string v(env);
  if (v == "debug") return LogLevel::kDebug;
  if (v == "warn") return LogLevel::kWarn;
  if (v == "error") return LogLevel::kError;
  return LogLevel::kInfo;
}

std::atomic<int> g_log_level{static_cast<int>(initial_log_level())};
std::mutex g_log_mutex;

}  // namespace

void set_log_level(LogLevel level) { g_log_level = static_cast<int>(level); }
LogLevel log_level() { return static_cast<LogLevel>(g_log_level.load()); }

void log_message(LogLevel level, const std::string& msg) {
  static const char* kNames[] = {"DEBUG", "INFO", "WARN", "ERROR"};
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::fprintf(stderr, "[redline %s] %s\n", kNames[static_cast<int>(level)], msg.c_str());
}

float half_bits_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {  // subnormal: renormalize
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t float_to_half_bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t abs = x & 0x7FFFFFFFu;
  if (abs >= 0x7F800000u) {  // inf / nan
    return static_cast<uint16_t>(sign | 0x7C00u | (abs > 0x7F800000u ? 0x200u : 0u));
  }
  if (abs >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow
  if (abs < 0x38800000u) {  // subnormal or zero in fp16
    if (abs < 0x33000000u) return static_cast<uint16_t>(sign);
    const uint32_t exp = abs >> 23;
    uint32_t mant = (abs & 0x7FFFFFu) | 0x800000u;
    const uint32_t shift = 126 - exp;  // 14 - (exp - 127) + 13 - 1
    const uint32_t rounded = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1);
    const uint32_t half_point = 1u << (shift - 1);
    uint32_t result = rounded;
    if (rem > half_point || (rem == half_point && (rounded & 1u))) ++result;
    return static_cast<uint16_t>(sign | result);
  }
  uint32_t result = ((abs >> 13) - (112u << 10));
  const uint32_t rem = abs & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (result & 1u))) ++result;
  return static_cast<uint16_t>(sign | result);
}

std::string format_bytes(uint64_t bytes) {
  char buf[64];
  const double b = static_cast<double>(bytes);
  if (b >= 1024.0 * 1024 * 1024) {
    std::snprintf(buf, sizeof(buf), "%.2f GiB", b / (1024.0 * 1024 * 1024));
  } else if (b >= 1024.0 * 1024) {
    std::snprintf(buf, sizeof(buf), "%.1f MiB", b / (1024.0 * 1024));
  } else if (b >= 1024.0) {
    std::snprintf(buf, sizeof(buf), "%.1f KiB", b / 1024.0);
  } else {
    std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
  }
  return buf;
}

}  // namespace redline
