#include "cuda/gemm/cublaslt_gemm.h"

#include "cuda/cuda_utils.h"
#include "cuda/kernels/kernels.h"

#define RL_CUBLAS_CHECK(expr)                                                               \
  do {                                                                                      \
    const cublasStatus_t rl_st_ = (expr);                                                   \
    if (rl_st_ != CUBLAS_STATUS_SUCCESS) {                                                  \
      throw ::redline::Error(::redline::str_cat(__FILE__, ":", __LINE__, ": ", #expr,       \
                                                " failed with status ", static_cast<int>(rl_st_))); \
    }                                                                                       \
  } while (0)

namespace redline {

namespace {
constexpr int kKindFp16 = 0;
constexpr int kKindInt8 = 1;
}  // namespace

CublasLtGemm::CublasLtGemm(size_t workspace_bytes, MemoryLedger* ledger)
    : workspace_(ledger->alloc("workspace", workspace_bytes)) {
  RL_CUBLAS_CHECK(cublasLtCreate(&handle_));
}

CublasLtGemm::~CublasLtGemm() {
  for (auto& kv : plans_) {
    Plan& p = kv.second;
    if (p.a) cublasLtMatrixLayoutDestroy(p.a);
    if (p.b) cublasLtMatrixLayoutDestroy(p.b);
    if (p.c) cublasLtMatrixLayoutDestroy(p.c);
    if (p.desc) cublasLtMatmulDescDestroy(p.desc);
  }
  if (handle_) cublasLtDestroy(handle_);
}

void CublasLtGemm::select_algo(Plan& plan, const char* what) {
  cublasLtMatmulPreference_t pref;
  RL_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  const size_t ws = workspace_.bytes();
  RL_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  cublasLtMatmulHeuristicResult_t result{};
  int found = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(handle_, plan.desc, plan.a, plan.b,
                                                           plan.c, plan.c, pref, 1, &result, &found);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS || found == 0) {
    throw Error(str_cat("cuBLASLt found no algorithm for ", what));
  }
  plan.algo = result.algo;
}

// Column-major view used by cuBLAS: D^T[n, m] = W[n, k] . X^T[k, m], i.e.
// A = W stored as col-major [k, n] (transposed), B = X as col-major [k, m].
CublasLtGemm::Plan& CublasLtGemm::plan_fp16(int m, int n, int k, int ldy, bool y_fp32, bool bias) {
  const Key key{kKindFp16 + (y_fp32 ? 10 : 0), m, n, k, ldy, bias ? 1 : 0};
  auto it = plans_.find(key);
  if (it != plans_.end()) return it->second;
  Plan p;
  RL_CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  RL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  RL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  if (bias) {
    const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    RL_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    const cudaDataType_t bias_type = CUDA_R_16F;
    RL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                   &bias_type, sizeof(bias_type)));
  }
  RL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, k, n, k));
  RL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, k, m, k));
  RL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.c, y_fp32 ? CUDA_R_32F : CUDA_R_16F, n, m, ldy));
  select_algo(p, "fp16 GEMM");
  return plans_.emplace(key, p).first->second;
}

CublasLtGemm::Plan& CublasLtGemm::plan_int8(int m, int n, int k) {
  const Key key{kKindInt8, m, n, k, n, 0};
  auto it = plans_.find(key);
  if (it != plans_.end()) return it->second;
  Plan p;
  RL_CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  RL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  RL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  RL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8I, k, n, k));
  RL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8I, k, m, k));
  RL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32I, n, m, n));
  select_algo(p, "int8 GEMM");
  return plans_.emplace(key, p).first->second;
}

void CublasLtGemm::gemm_fp16(const half* x, const half* w, const half* bias, void* y, bool y_fp32,
                             int m, int n, int k, int ldy, cudaStream_t stream) {
  if (m <= 0) return;
  // The bias epilogue needs bias and D of the same precision, so fp32 outputs add it separately.
  const bool bias_epilogue = bias != nullptr && !y_fp32;
  Plan& p = plan_fp16(m, n, k, ldy, y_fp32, bias_epilogue);
  if (bias_epilogue) {
    RL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                                   sizeof(bias)));
  }
  const float alpha = 1.f, beta = 0.f;
  RL_CUBLAS_CHECK(cublasLtMatmul(handle_, p.desc, &alpha, w, p.a, x, p.b, &beta, y, p.c, y, p.c,
                                 &p.algo, workspace_.get(), workspace_.bytes(), stream));
  if (bias != nullptr && y_fp32) kernels::add_bias_f32(static_cast<float*>(y), bias, m, n, ldy, stream);
}

void CublasLtGemm::gemm_int8(const int8_t* x, const int8_t* w, int32_t* y, int m, int n, int k,
                             cudaStream_t stream) {
  if (m <= 0) return;
  RL_CHECK(k % 4 == 0, "int8 GEMM needs K % 4 == 0");
  Plan& p = plan_int8(m, n, k);
  const int32_t alpha = 1, beta = 0;
  RL_CUBLAS_CHECK(cublasLtMatmul(handle_, p.desc, &alpha, w, p.a, x, p.b, &beta, y, p.c, y, p.c,
                                 &p.algo, workspace_.get(), workspace_.bytes(), stream));
}

}  // namespace redline
