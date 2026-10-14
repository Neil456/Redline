// cuBLASLt GEMMs with cached descriptors and heuristic algorithm selection.
// All calls use a preallocated workspace so they can be captured in CUDA graphs.
#pragma once

#include <cublasLt.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <map>
#include <tuple>

#include "cuda/device_memory.h"

namespace redline {

class CublasLtGemm {
 public:
  CublasLtGemm(size_t workspace_bytes, MemoryLedger* ledger);
  ~CublasLtGemm();
  CublasLtGemm(const CublasLtGemm&) = delete;
  CublasLtGemm& operator=(const CublasLtGemm&) = delete;

  // y[m, n] = x[m, k] . w[n, k]^T (+ bias[n]). x and w are fp16 (row-major,
  // leading dims k); y is fp16 or fp32 with row stride ldy (>= n), which lets
  // callers write a column slice of a wider output.
  void gemm_fp16(const half* x, const half* w, const half* bias, void* y, bool y_fp32, int m,
                 int n, int k, int ldy, cudaStream_t stream);

  // y[m, n] (int32) = x[m, k] (int8) . w[n, k]^T (int8). Requires k % 4 == 0.
  void gemm_int8(const int8_t* x, const int8_t* w, int32_t* y, int m, int n, int k,
                 cudaStream_t stream);

 private:
  struct Plan {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
  };
  using Key = std::tuple<int, int, int, int, int, int>;  // kind, m, n, k, ldy, epilogue

  Plan& plan_fp16(int m, int n, int k, int ldy, bool y_fp32, bool bias);
  Plan& plan_int8(int m, int n, int k);
  void select_algo(Plan& plan, const char* what);

  cublasLtHandle_t handle_ = nullptr;
  DeviceBuffer workspace_;
  std::map<Key, Plan> plans_;
};

}  // namespace redline
