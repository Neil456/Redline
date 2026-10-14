#include "cuda/gemm/linear.h"

#include <algorithm>
#include <cstdlib>

#include "cuda/cuda_utils.h"
#include "cuda/kernels/kernels.h"

namespace redline {

const char* prefill_gemm_name(PrefillGemm g) {
  switch (g) {
    case PrefillGemm::kAuto: return "auto";
    case PrefillGemm::kDequant: return "dequant";
    case PrefillGemm::kW8A8: return "w8a8";
    case PrefillGemm::kCutlass: return "cutlass";
    case PrefillGemm::kTrtllm: return "trtllm";
  }
  return "?";
}

PrefillGemm prefill_gemm_from_name(const std::string& s) {
  if (s == "auto") return PrefillGemm::kAuto;
  if (s == "dequant" || s == "cublaslt") return PrefillGemm::kDequant;
  if (s == "w8a8") return PrefillGemm::kW8A8;
  if (s == "cutlass") return PrefillGemm::kCutlass;
  if (s == "trtllm") return PrefillGemm::kTrtllm;
  throw Error("unknown prefill GEMM backend: " + s);
}

bool LinearExecutor::cutlass_available() {
#ifdef REDLINE_WITH_CUTLASS
  return true;
#else
  return false;
#endif
}

bool LinearExecutor::trtllm_available() {
#ifdef REDLINE_WITH_TRTLLM
  return true;
#else
  return false;
#endif
}

LinearExecutor::LinearExecutor(const LinearOptions& opts, MemoryLedger* ledger)
    : opts_(opts), lt_(opts.cublaslt_workspace, ledger) {
  RL_CHECK(opts_.gemv_max_m >= 0 && opts_.gemv_max_m <= kernels::kGemvMaxM,
           "gemv_max_m must be in [0, ", kernels::kGemvMaxM, "]");
  // Resolve "auto" to the best accuracy-preserving backend in this build.
  int8_backend_ = opts_.int8_prefill;
  if (int8_backend_ == PrefillGemm::kAuto) {
    int8_backend_ = cutlass_available() ? PrefillGemm::kCutlass : PrefillGemm::kDequant;
  }
  int4_backend_ = opts_.int4_prefill;
  if (int4_backend_ == PrefillGemm::kAuto) int4_backend_ = PrefillGemm::kDequant;
  RL_CHECK(int8_backend_ != PrefillGemm::kCutlass || cutlass_available(),
           "CUTLASS backend requested but Redline was built without REDLINE_WITH_CUTLASS");
  RL_CHECK(int8_backend_ != PrefillGemm::kTrtllm || trtllm_available(),
           "TensorRT-LLM backend requested but Redline was built without REDLINE_WITH_TRTLLM");
  RL_CHECK(int4_backend_ != PrefillGemm::kTrtllm || trtllm_available(),
           "TensorRT-LLM backend requested but Redline was built without REDLINE_WITH_TRTLLM");
  RL_CHECK(int4_backend_ == PrefillGemm::kDequant || int4_backend_ == PrefillGemm::kTrtllm,
           "INT4 prefill supports the dequant and trtllm backends");

  dequant_scratch_ = ledger->alloc("workspace", opts_.dequant_scratch);
  if (opts_.int4_tensor_cores) {
    // Split-K partials: splits * n <= target CTAs * 256 + n for every layer (see
    // kernels::w4a16_tc_splits), so this bounds every INT4 GEMM up to max_n columns.
    int sms = 0, dev = 0;
    RL_CUDA_CHECK(cudaGetDevice(&dev));
    RL_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const size_t max_n = static_cast<size_t>(std::max(opts_.max_n, 256));
    const char* e = std::getenv("REDLINE_W4TC_CTAS_PER_SM");
    const size_t per_sm = static_cast<size_t>(std::max(2, e != nullptr ? std::atoi(e) : 2));
    const size_t floats = static_cast<size_t>(kernels::kW4TcMaxM) * (per_sm * sms * 256u + max_n);
    tc_partials_ = ledger->alloc("workspace", floats * sizeof(float));
    const size_t tickets = max_n / 256 + 64;
    tc_tickets_ = ledger->alloc("workspace", tickets * sizeof(int));
    RL_CUDA_CHECK(cudaMemset(tc_tickets_.get(), 0, tc_tickets_.bytes()));
  }
  if (int8_backend_ == PrefillGemm::kW8A8) {
    RL_CHECK(opts_.max_k > 0 && opts_.max_n > 0, "W8A8 needs max_k / max_n");
    act_q_ = ledger->alloc("workspace", static_cast<size_t>(opts_.max_m) * opts_.max_k);
    act_scales_ = ledger->alloc("workspace", static_cast<size_t>(opts_.max_m) * sizeof(float));
    acc_i32_ = ledger->alloc("workspace",
                             static_cast<size_t>(opts_.max_m) * opts_.max_n * sizeof(int32_t));
  }
#ifdef REDLINE_WITH_CUTLASS
  if (int8_backend_ == PrefillGemm::kCutlass) cutlass_ = make_cutlass_mixed_gemm(ledger);
#endif
#ifdef REDLINE_WITH_TRTLLM
  if (int8_backend_ == PrefillGemm::kTrtllm || int4_backend_ == PrefillGemm::kTrtllm) {
    trtllm_ = make_trtllm_gemm(ledger, opts_.max_m);
  }
#endif
}

LinearExecutor::~LinearExecutor() = default;

std::string LinearExecutor::describe() const {
  return str_cat("gemv_max_m=", opts_.gemv_max_m, " int8_prefill=", prefill_gemm_name(int8_backend_),
                 " int4_prefill=", prefill_gemm_name(int4_backend_),
                 " int4_kernel=", opts_.int4_tensor_cores ? "tensor-core" : "gemv");
}

void LinearExecutor::forward(const LinearWeight& w, const half* x, int m, void* y, bool y_fp32,
                             cudaStream_t stream) {
  if (m <= 0) return;
  RL_CHECK(m <= opts_.max_m || w.qtype == QuantType::kFP16, "linear ", w.name, ": m=", m,
           " exceeds max_m=", opts_.max_m);
  if (w.qtype == QuantType::kFP16) {
    lt_.gemm_fp16(x, static_cast<const half*>(w.weight), w.bias, y, y_fp32, m, w.n, w.k, w.n,
                  stream);
    return;
  }
  if (w.trtllm_layout) {
    RL_CHECK(trtllm_ && !y_fp32, "linear ", w.name, " needs the TensorRT-LLM backend (fp16 out)");
    trtllm_->run(w, x, m, static_cast<half*>(y), stream);
    return;
  }
  if (w.int4_tc) {
    if (m <= kernels::kW4TcMaxM) {
      tc_gemm(w, x, m, y, y_fp32, false, stream);
    } else {
      dequant_gemm(w, x, m, y, y_fp32, stream);
    }
    return;
  }
  if (m <= opts_.gemv_max_m) {
    gemv(w, x, m, y, y_fp32, stream);
    return;
  }
  const bool int8_tensor_core = w.qtype == QuantType::kINT8 && !y_fp32 &&
                                ((int8_backend_ == PrefillGemm::kCutlass && cutlass_) ||
                                 int8_backend_ == PrefillGemm::kW8A8);
  if (!int8_tensor_core && m <= opts_.gemv_chunked_max_m && opts_.gemv_max_m > 0) {
    // Batched decode: re-streaming quantized weights a few times is still
    // cheaper than materializing an fp16 copy of the layer.
    const size_t elem = y_fp32 ? sizeof(float) : sizeof(half);
    const int chunk = w.qtype == QuantType::kINT4 ? kernels::kGemvW4MaxM : opts_.gemv_max_m;
    for (int r0 = 0; r0 < m; r0 += chunk) {
      const int rows = std::min(chunk, m - r0);
      gemv(w, x + static_cast<size_t>(r0) * w.k, rows,
           static_cast<char*>(y) + static_cast<size_t>(r0) * w.n * elem, y_fp32, stream);
    }
    return;
  }
  if (int8_tensor_core && int8_backend_ == PrefillGemm::kCutlass) {
    half* yh = static_cast<half*>(y);
    cutlass_->run(x, static_cast<const int8_t*>(w.weight), yh, m, w.n, w.k, stream);
    kernels::scale_columns(yh, w.scales, CutlassMixedGemm::kOutputDivisor, w.bias, m, w.n, stream);
    return;
  }
  if (int8_tensor_core && int8_backend_ == PrefillGemm::kW8A8) {
    w8a8_gemm(w, x, m, static_cast<half*>(y), stream);
    return;
  }
  dequant_gemm(w, x, m, y, y_fp32, stream);
}

void LinearExecutor::forward_gated(const LinearWeight& w, const half* x, int m, half* out,
                                   half* scratch, cudaStream_t stream) {
  if (m <= 0) return;
  const int inter = w.n / 2;
  if (w.int4_tc && w.interleaved && m <= kernels::kW4TcMaxM && w.bias == nullptr) {
    tc_gemm(w, x, m, out, false, /*silu_mul=*/true, stream);
    return;
  }
  forward(w, x, m, scratch, false, stream);
  if (w.interleaved) {
    kernels::silu_and_mul_interleaved(scratch, m, inter, out, stream);
  } else {
    kernels::silu_and_mul(scratch, m, inter, out, stream);
  }
}

void LinearExecutor::tc_gemm(const LinearWeight& w, const half* x, int m, void* y, bool y_fp32,
                             bool silu_mul, cudaStream_t stream) {
  RL_CHECK(tc_partials_.get() != nullptr, "linear ", w.name,
           " is in the tensor-core INT4 layout but the executor has no workspace");
  kernels::W4TcWorkspace ws;
  ws.partials = tc_partials_.as<float>();
  ws.partial_floats = tc_partials_.bytes() / sizeof(float);
  ws.tickets = tc_tickets_.as<int>();
  ws.num_tickets = static_cast<int>(tc_tickets_.bytes() / sizeof(int));
  kernels::gemm_w4a16_tc(x, m, static_cast<const uint8_t*>(w.weight), w.scales, w.offsets,
                         w.group_size, silu_mul ? nullptr : w.bias, w.n, w.k, y, y_fp32, silu_mul,
                         ws, stream);
}

void LinearExecutor::gemv(const LinearWeight& w, const half* x, int m, void* y, bool y_fp32,
                          cudaStream_t stream) {
  if (w.qtype == QuantType::kINT8) {
    kernels::gemv_w8a16(x, m, static_cast<const int8_t*>(w.weight), w.scales, w.bias, w.n, w.k, y,
                        y_fp32, stream);
  } else {
    kernels::gemv_w4a16(x, m, static_cast<const uint8_t*>(w.weight), w.scales, w.offsets,
                        w.group_size, w.bias, w.n, w.k, y, y_fp32, stream);
  }
}

// Dequantizes row tiles of W into the fp16 scratch buffer and runs a cuBLASLt
// GEMM per tile, writing the matching column slice of y.
void LinearExecutor::dequant_gemm(const LinearWeight& w, const half* x, int m, void* y,
                                  bool y_fp32, cudaStream_t stream) {
  const size_t row_bytes = static_cast<size_t>(w.k) * sizeof(half);
  int tile = static_cast<int>(std::min<size_t>(dequant_scratch_.bytes() / row_bytes, 32768));
  tile = std::max(64, tile - tile % 64);
  RL_CHECK(static_cast<size_t>(tile) * row_bytes <= dequant_scratch_.bytes(),
           "dequant scratch too small for K=", w.k);
  half* scratch = dequant_scratch_.as<half>();
  const size_t elem = y_fp32 ? sizeof(float) : sizeof(half);
  for (int n0 = 0; n0 < w.n; n0 += tile) {
    const int n1 = std::min(w.n, n0 + tile);
    if (w.qtype == QuantType::kINT8) {
      kernels::dequantize_weight_int8(static_cast<const int8_t*>(w.weight), w.scales, w.k, n0, n1,
                                      scratch, stream);
    } else if (w.int4_tc) {
      kernels::dequantize_weight_int4_tc(static_cast<const uint8_t*>(w.weight), w.scales, w.offsets,
                                         w.k, w.group_size, n0, n1, scratch, stream);
    } else {
      kernels::dequantize_weight_int4(static_cast<const uint8_t*>(w.weight), w.scales, w.offsets,
                                      w.k, w.group_size, n0, n1, scratch, stream);
    }
    void* y_slice = static_cast<char*>(y) + static_cast<size_t>(n0) * elem;
    lt_.gemm_fp16(x, scratch, w.bias ? w.bias + n0 : nullptr, y_slice, y_fp32, m, n1 - n0, w.k,
                  w.n, stream);
  }
}

void LinearExecutor::w8a8_gemm(const LinearWeight& w, const half* x, int m, half* y,
                               cudaStream_t stream) {
  RL_CHECK(w.k <= opts_.max_k && w.n <= opts_.max_n, "W8A8 buffers too small for ", w.name);
  int8_t* xq = act_q_.as<int8_t>();
  float* xs = act_scales_.as<float>();
  int32_t* acc = acc_i32_.as<int32_t>();
  kernels::quantize_rows_int8(x, m, w.k, xq, xs, stream);
  lt_.gemm_int8(xq, static_cast<const int8_t*>(w.weight), acc, m, w.n, w.k, stream);
  kernels::dequantize_int32_gemm_output(acc, xs, w.scales, w.bias, m, w.n, y, stream);
}

}  // namespace redline
