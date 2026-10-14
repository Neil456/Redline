// Optional third-party GEMM backends. Interfaces are always visible; the
// implementations are compiled only with REDLINE_WITH_CUTLASS /
// REDLINE_WITH_TRTLLM, and the factories are only called in those builds.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>

#include "cuda/device_memory.h"

namespace redline {

struct LinearWeight;

// CUTLASS 2.x GemmUniversal with OpMultiplyAddMixedInputUpcast: fp16
// activations x int8 weights on fp16 tensor cores (sm80+), fp32 accumulate.
// The per-channel scale is applied afterwards (kernels::scale_columns), so the
// kernel emits acc / kOutputDivisor to stay inside the fp16 range.
class CutlassMixedGemm {
 public:
  static constexpr float kOutputDivisor = 128.f;
  virtual ~CutlassMixedGemm() = default;
  // y[m, n] = (x[m, k] . q[n, k]^T) / kOutputDivisor
  virtual void run(const half* x, const int8_t* q, half* y, int m, int n, int k,
                   cudaStream_t stream) = 0;
};
std::unique_ptr<CutlassMixedGemm> make_cutlass_mixed_gemm(MemoryLedger* ledger);

// TensorRT-LLM CutlassFpAIntBGemmRunner (fine-grained scales + zeros for
// INT4, per-channel scales for INT8). Weights are converted once at load time
// into TensorRT-LLM's interleaved layout; when this backend is enabled the
// converted copy replaces Redline's layout and serves every batch size.
class TrtllmWeightOnlyGemm {
 public:
  virtual ~TrtllmWeightOnlyGemm() = default;
  // Converts a Redline-format quantized linear (host pointers) into device
  // buffers in TensorRT-LLM layout and points `w` at them.
  virtual void prepare(LinearWeight* w, const void* host_weight, const uint16_t* host_scales,
                       const uint16_t* host_offsets, MemoryLedger* ledger) = 0;
  virtual void run(const LinearWeight& w, const half* x, int m, half* y, cudaStream_t stream) = 0;
};
std::unique_ptr<TrtllmWeightOnlyGemm> make_trtllm_gemm(MemoryLedger* ledger, int max_m);

}  // namespace redline
