// Linear layers: weight representation and the GEMM dispatcher that picks a
// kernel per (quantization, batch rows) combination.
//
//   weights  rows m <= gemv_max_m (decode)   rows m > gemv_max_m (batched decode / prefill)
//   FP16     cuBLASLt                         cuBLASLt
//   INT8     W8A16 GEMV kernel                CUTLASS mixed-input GEMM | chunked GEMV (m <= 32)
//                                             | cuBLASLt W8A8 | dequantize + cuBLASLt
//   INT4     W4A16 GEMV kernel                TensorRT-LLM fpA_intB | GEMV in 16-row chunks
//                                             (m <= 32) | dequantize + cuBLASLt
#pragma once

#include <cuda_fp16.h>

#include <memory>
#include <string>

#include "core/model_config.h"
#include "cuda/device_memory.h"
#include "cuda/gemm/backends.h"
#include "cuda/gemm/cublaslt_gemm.h"

namespace redline {

struct LinearWeight {
  std::string name;
  QuantType qtype = QuantType::kFP16;
  int n = 0;  // output features
  int k = 0;  // input features
  int group_size = 0;
  const void* weight = nullptr;  // half [n,k] | int8 [n,k] | uint8 [n,k/2]
  const half* scales = nullptr;  // [n] (INT8) or [n, k/g] (INT4)
  const half* offsets = nullptr; // [n, k/g] (INT4)
  const half* bias = nullptr;    // [n] or null
  // Set when the weight was converted for the TensorRT-LLM backend: `weight`,
  // `scales` and `trt_zeros` then use TensorRT-LLM's layout.
  bool trtllm_layout = false;
  const half* trt_zeros = nullptr;
};

enum class PrefillGemm {
  kAuto,      // best available accuracy-preserving backend
  kDequant,   // dequantize weight tiles to fp16, then cuBLASLt
  kW8A8,      // dynamic per-token INT8 activations + cuBLASLt INT8 tensor cores
  kCutlass,   // CUTLASS FP16 x INT8 mixed-input tensor-core GEMM
  kTrtllm,    // TensorRT-LLM fpA_intB weight-only GEMM
};
const char* prefill_gemm_name(PrefillGemm g);
PrefillGemm prefill_gemm_from_name(const std::string& s);

struct LinearOptions {
  int gemv_max_m = 8;              // use the weight-only GEMV kernels up to this many rows
  int gemv_chunked_max_m = 32;     // beyond that, chunked GEMV (8 rows INT8, 16 rows INT4) up to here;
                                   // measured on an RTX 4090: past 32 rows dequant + cuBLASLt wins
                                   // (unless a tensor-core mixed-input GEMM is available)
  PrefillGemm int8_prefill = PrefillGemm::kAuto;
  PrefillGemm int4_prefill = PrefillGemm::kAuto;
  size_t cublaslt_workspace = 32u << 20;
  size_t dequant_scratch = 64u << 20;  // fp16 scratch for dequantized weight tiles
  int max_m = 2048;                    // largest row count passed to forward()
  int max_k = 0;                       // largest K (needed for W8A8 buffers)
  int max_n = 0;                       // largest N (needed for W8A8 buffers)
};

class LinearExecutor {
 public:
  LinearExecutor(const LinearOptions& opts, MemoryLedger* ledger);
  ~LinearExecutor();

  // y[m, n] = x[m, k] . W^T (+ bias); y is fp32 when y_fp32 else fp16.
  void forward(const LinearWeight& w, const half* x, int m, void* y, bool y_fp32,
               cudaStream_t stream);

  const LinearOptions& options() const { return opts_; }
  PrefillGemm resolved_int8_prefill() const { return int8_backend_; }
  PrefillGemm resolved_int4_prefill() const { return int4_backend_; }
  TrtllmWeightOnlyGemm* trtllm() { return trtllm_.get(); }
  std::string describe() const;

  static bool cutlass_available();
  static bool trtllm_available();

 private:
  void gemv(const LinearWeight& w, const half* x, int m, void* y, bool y_fp32,
            cudaStream_t stream);
  void dequant_gemm(const LinearWeight& w, const half* x, int m, void* y, bool y_fp32,
                    cudaStream_t stream);
  void w8a8_gemm(const LinearWeight& w, const half* x, int m, half* y, cudaStream_t stream);

  LinearOptions opts_;
  PrefillGemm int8_backend_;
  PrefillGemm int4_backend_;
  CublasLtGemm lt_;
  DeviceBuffer dequant_scratch_;
  DeviceBuffer act_q_, act_scales_, acc_i32_;  // W8A8 path
  std::unique_ptr<CutlassMixedGemm> cutlass_;
  std::unique_ptr<TrtllmWeightOnlyGemm> trtllm_;
};

}  // namespace redline
