// Embedding lookup, fused residual-add + RMSNorm, SwiGLU, row gathers.
#include <algorithm>

#include "cuda/cuda_utils.h"
#include "cuda/kernels/common.cuh"
#include "cuda/kernels/kernels.h"

namespace redline {
namespace kernels {
namespace {

__global__ void resolve_input_ids_kernel(int32_t* ids, const int32_t* src,
                                         const int32_t* prev_sampled, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && src[i] >= 0) ids[i] = prev_sampled[src[i]];
}

// One block per token; hidden must be a multiple of 8 (16-byte vectors).
__global__ void embedding_kernel(const int32_t* ids, const half* table, int hidden, half* out) {
  const int t = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(table + static_cast<size_t>(ids[t]) * hidden);
  uint4* dst = reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * hidden);
  for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x) dst[i] = src[i];
}

// One block per row. Follows HF Llama numerics in fp16: the residual sum is
// rounded to fp16, the variance is computed in fp32, the normalized value is
// rounded to fp16 and then multiplied by the (fp16) weight.
__global__ void fused_add_rmsnorm_kernel(half* residual, const half* delta, const half* weight,
                                         int hidden, float eps, half* out) {
  __shared__ float scratch[32];
  const size_t row = static_cast<size_t>(blockIdx.x) * hidden;
  uint4* res = reinterpret_cast<uint4*>(residual + row);
  const uint4* del = delta ? reinterpret_cast<const uint4*>(delta + row) : nullptr;
  const int nvec = hidden / 8;

  float ss = 0.f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    float r[8];
    unpack_half8(res[i], r);
    if (del) {
      float d[8];
      unpack_half8(del[i], d);
#pragma unroll
      for (int j = 0; j < 8; ++j) r[j] = __half2float(__float2half_rn(r[j] + d[j]));
      res[i] = pack_half8(r);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) ss += r[j] * r[j];
  }
  ss = block_reduce_sum(ss, scratch);
  const float inv = rsqrtf(ss / hidden + eps);

  const uint4* w = reinterpret_cast<const uint4*>(weight);
  uint4* o = reinterpret_cast<uint4*>(out + row);
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    float r[8], wf[8], y[8];
    unpack_half8(res[i], r);
    unpack_half8(w[i], wf);
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = __half2float(__float2half_rn(r[j] * inv)) * wf[j];
    o[i] = pack_half8(y);
  }
}

__global__ void silu_and_mul_kernel(const half* gate_up, int inter, half* out) {
  const size_t row = blockIdx.x;
  const uint4* g = reinterpret_cast<const uint4*>(gate_up + row * 2 * inter);
  const uint4* u = reinterpret_cast<const uint4*>(gate_up + row * 2 * inter + inter);
  uint4* o = reinterpret_cast<uint4*>(out + row * inter);
  for (int i = threadIdx.x; i < inter / 8; i += blockDim.x) {
    float gf[8], uf[8], y[8];
    unpack_half8(g[i], gf);
    unpack_half8(u[i], uf);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float s = __half2float(__float2half_rn(gf[j] / (1.f + __expf(-gf[j]))));
      y[j] = s * uf[j];
    }
    o[i] = pack_half8(y);
  }
}

__global__ void gather_rows_kernel(const half* src, const int32_t* idx, int cols, half* dst) {
  const int r = blockIdx.x;
  const uint4* s = reinterpret_cast<const uint4*>(src + static_cast<size_t>(idx[r]) * cols);
  uint4* d = reinterpret_cast<uint4*>(dst + static_cast<size_t>(r) * cols);
  for (int i = threadIdx.x; i < cols / 8; i += blockDim.x) d[i] = s[i];
}

int row_threads(int vecs) {
  int t = 32;
  while (t < vecs && t < 1024) t *= 2;
  return t;
}

}  // namespace

void resolve_input_ids(int32_t* input_ids, const int32_t* src, const int32_t* prev_sampled, int n,
                       cudaStream_t stream) {
  if (n <= 0) return;
  resolve_input_ids_kernel<<<(n + 255) / 256, 256, 0, stream>>>(input_ids, src, prev_sampled, n);
  RL_CUDA_CHECK_LAUNCH();
}

void embedding_lookup(const int32_t* ids, const half* table, int n, int hidden, half* out,
                      cudaStream_t stream) {
  if (n <= 0) return;
  RL_CHECK(hidden % 8 == 0, "hidden size must be a multiple of 8");
  embedding_kernel<<<n, std::min(row_threads(hidden / 8), 256), 0, stream>>>(ids, table, hidden, out);
  RL_CUDA_CHECK_LAUNCH();
}

void fused_add_rmsnorm(half* residual, const half* delta, const half* weight, int rows, int hidden,
                       float eps, half* out, cudaStream_t stream) {
  if (rows <= 0) return;
  RL_CHECK(hidden % 8 == 0, "hidden size must be a multiple of 8");
  const int threads = std::min(row_threads(hidden / 8), 512);
  fused_add_rmsnorm_kernel<<<rows, threads, 0, stream>>>(residual, delta, weight, hidden, eps, out);
  RL_CUDA_CHECK_LAUNCH();
}

void silu_and_mul(const half* gate_up, int rows, int inter, half* out, cudaStream_t stream) {
  if (rows <= 0) return;
  RL_CHECK(inter % 8 == 0, "intermediate size must be a multiple of 8");
  silu_and_mul_kernel<<<rows, std::min(row_threads(inter / 8), 512), 0, stream>>>(gate_up, inter, out);
  RL_CUDA_CHECK_LAUNCH();
}

void gather_rows(const half* src, const int32_t* idx, int n, int cols, half* dst,
                 cudaStream_t stream) {
  if (n <= 0) return;
  RL_CHECK(cols % 8 == 0, "row width must be a multiple of 8");
  gather_rows_kernel<<<n, std::min(row_threads(cols / 8), 256), 0, stream>>>(src, idx, cols, dst);
  RL_CUDA_CHECK_LAUNCH();
}

}  // namespace kernels
}  // namespace redline
