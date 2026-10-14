// Helpers for GPU tests: device vectors, random data, fp16 conversion.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <sstream>
#include <random>
#include <vector>

#include "core/common.h"
#include "cuda/cuda_utils.h"
#include "test_framework.h"

namespace rltest {

template <typename T>
class DeviceVec {
 public:
  DeviceVec() = default;
  explicit DeviceVec(size_t n) : n_(n) { RL_CUDA_CHECK(cudaMalloc(&p_, std::max<size_t>(n, 1) * sizeof(T))); }
  explicit DeviceVec(const std::vector<T>& h) : DeviceVec(h.size()) { upload(h); }
  ~DeviceVec() {
    if (p_) cudaFree(p_);
  }
  DeviceVec(const DeviceVec&) = delete;
  DeviceVec& operator=(const DeviceVec&) = delete;
  DeviceVec(DeviceVec&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; }
  DeviceVec& operator=(DeviceVec&& o) noexcept {
    if (this != &o) {
      if (p_) cudaFree(p_);
      p_ = o.p_;
      n_ = o.n_;
      o.p_ = nullptr;
    }
    return *this;
  }

  void upload(const std::vector<T>& h) {
    RL_CUDA_CHECK(cudaMemcpy(p_, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> h(n_);
    RL_CUDA_CHECK(cudaDeviceSynchronize());
    RL_CUDA_CHECK(cudaMemcpy(h.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  void zero() { RL_CUDA_CHECK(cudaMemset(p_, 0, n_ * sizeof(T))); }
  T* get() const { return p_; }
  size_t size() const { return n_; }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

inline std::vector<float> randn(size_t n, float stddev, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> d(0.f, stddev);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

inline std::vector<half> to_half(const std::vector<float>& v) {
  std::vector<half> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half_rn(v[i]);
  return h;
}

inline std::vector<float> to_float(const std::vector<half>& h) {
  std::vector<float> v(h.size());
  for (size_t i = 0; i < h.size(); ++i) v[i] = __half2float(h[i]);
  return v;
}

// Rounds through fp16 (what the GPU sees).
inline std::vector<float> round_half(const std::vector<float>& v) { return to_float(to_half(v)); }

// max |a - b| / (max |b| + eps)
inline double rel_max_err(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, static_cast<double>(std::fabs(a[i] - b[i])));
    den = std::max(den, static_cast<double>(std::fabs(b[i])));
  }
  return num / (den + 1e-12);
}

inline bool gpu_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

}  // namespace rltest

#define EXPECT_CLOSE_VEC(a, b, tol)                                                    \
  do {                                                                                 \
    const double e_ = ::rltest::rel_max_err((a), (b));                                 \
    if (!(e_ <= (tol))) {                                                              \
      std::ostringstream os_;                                                          \
      os_ << __FILE__ << ":" << __LINE__ << ": relative max error " << e_ << " > " << (tol); \
      throw ::rltest::Failure{os_.str()};                                              \
    }                                                                                  \
  } while (0)
