// CUDA error checking and small device helpers shared by host and device code.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core/common.h"

#define RL_CUDA_CHECK(expr)                                                              \
  do {                                                                                   \
    const cudaError_t rl_err_ = (expr);                                                  \
    if (rl_err_ != cudaSuccess) {                                                        \
      throw ::redline::Error(::redline::str_cat(__FILE__, ":", __LINE__, ": ", #expr,    \
                                                " failed: ", cudaGetErrorString(rl_err_))); \
    }                                                                                    \
  } while (0)

// Checks for launch configuration errors without synchronizing.
#define RL_CUDA_CHECK_LAUNCH() RL_CUDA_CHECK(cudaGetLastError())

namespace redline {

// RAII CUDA event pair for timing a region on a stream.
class CudaTimer {
 public:
  CudaTimer() {
    RL_CUDA_CHECK(cudaEventCreate(&start_));
    RL_CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  CudaTimer(const CudaTimer&) = delete;
  CudaTimer& operator=(const CudaTimer&) = delete;

  void start(cudaStream_t s) { RL_CUDA_CHECK(cudaEventRecord(start_, s)); }
  void stop(cudaStream_t s) { RL_CUDA_CHECK(cudaEventRecord(stop_, s)); }
  // Blocks until the stop event completes.
  float elapsed_ms() {
    RL_CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    RL_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{}, stop_{};
};

}  // namespace redline
