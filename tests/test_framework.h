// Tiny self-contained test framework (no external dependencies).
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace rltest {

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct Failure {
  std::string msg;
};

inline int run_all(int argc, char** argv) {
  const std::string filter = argc > 1 ? argv[1] : "";
  int passed = 0, failed = 0, skipped = 0;
  for (const auto& t : registry()) {
    if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) {
      ++skipped;
      continue;
    }
    try {
      t.fn();
      std::printf("[ PASS ] %s\n", t.name);
      ++passed;
    } catch (const Failure& f) {
      std::printf("[ FAIL ] %s\n         %s\n", t.name, f.msg.c_str());
      ++failed;
    } catch (const std::exception& e) {
      std::printf("[ FAIL ] %s\n         exception: %s\n", t.name, e.what());
      ++failed;
    }
  }
  std::printf("\n%d passed, %d failed, %d filtered\n", passed, failed, skipped);
  return failed == 0 ? 0 : 1;
}

}  // namespace rltest

#define RL_TEST_CONCAT2(a, b) a##b
#define RL_TEST_CONCAT(a, b) RL_TEST_CONCAT2(a, b)
#define TEST(name)                                                                  \
  static void name();                                                               \
  static ::rltest::Registrar RL_TEST_CONCAT(registrar_, name)(#name, name);         \
  static void name()

#define EXPECT_TRUE(cond)                                                            \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::ostringstream os_;                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": expected true: " #cond;               \
      throw ::rltest::Failure{os_.str()};                                            \
    }                                                                                \
  } while (0)

#define EXPECT_EQ(a, b)                                                              \
  do {                                                                               \
    const auto va_ = (a);                                                            \
    const auto vb_ = (b);                                                            \
    if (!(va_ == vb_)) {                                                             \
      std::ostringstream os_;                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": " #a " == " #b " failed (" << va_     \
          << " vs " << vb_ << ")";                                                   \
      throw ::rltest::Failure{os_.str()};                                            \
    }                                                                                \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                                       \
  do {                                                                               \
    const double va_ = (a);                                                          \
    const double vb_ = (b);                                                          \
    if (!(std::fabs(va_ - vb_) <= (tol))) {                                          \
      std::ostringstream os_;                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": |" #a " - " #b "| <= " #tol           \
          << " failed (" << va_ << " vs " << vb_ << ")";                             \
      throw ::rltest::Failure{os_.str()};                                            \
    }                                                                                \
  } while (0)

#define EXPECT_THROWS(stmt)                                                          \
  do {                                                                               \
    bool threw_ = false;                                                             \
    try {                                                                            \
      stmt;                                                                          \
    } catch (...) {                                                                  \
      threw_ = true;                                                                 \
    }                                                                                \
    if (!threw_) {                                                                   \
      std::ostringstream os_;                                                        \
      os_ << __FILE__ << ":" << __LINE__ << ": expected exception from " #stmt;      \
      throw ::rltest::Failure{os_.str()};                                            \
    }                                                                                \
  } while (0)

#define RL_TEST_MAIN() \
  int main(int argc, char** argv) { return ::rltest::run_all(argc, argv); }
