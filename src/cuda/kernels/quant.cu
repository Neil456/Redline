// Quantization helpers: dynamic activation quantization for the W8A8
// cuBLASLt path, epilogues for integer / CUTLASS GEMM outputs, and weight
// dequantization for the dequantize-then-GEMM prefill fallback.
#include <algorithm>

#include "cuda/cuda_utils.h"
#include "cuda/kernels/common.cuh"
#include "cuda/kernels/kernels.h"

namespace redline {
namespace kernels {
namespace {

__global__ void quantize_rows_int8_kernel(const half* x, int cols, int8_t* q, float* scales) {
  __shared__ float scratch[32];
  const size_t row = blockIdx.x;
  const half* xr = x + row * cols;
  float amax = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) amax = fmaxf(amax, fabsf(__half2float(xr[i])));
  amax = block_reduce_max(amax, scratch);
  const float scale = amax > 0.f ? amax / 127.f : 1.f;
  const float inv = 1.f / scale;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    q[row * cols + i] = static_cast<int8_t>(__float2int_rn(__half2float(xr[i]) * inv));
  }
  if (threadIdx.x == 0) scales[row] = scale;
}

__global__ void dequant_i32_kernel(const int32_t* acc, const float* row_scales,
                                   const half* col_scales, const half* bias, int cols, half* out) {
  const size_t row = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  float v = static_cast<float>(acc[row * cols + c]) * row_scales[row] * __half2float(col_scales[c]);
  if (bias) v += __half2float(bias[c]);
  out[row * cols + c] = __float2half_rn(v);
}

__global__ void scale_columns_kernel(half* y, const half* col_scales, float mult, const half* bias,
                                     int cols) {
  const size_t row = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  float v = __half2float(y[row * cols + c]) * __half2float(col_scales[c]) * mult;
  if (bias) v += __half2float(bias[c]);
  y[row * cols + c] = __float2half_rn(v);
}

__global__ void dequant_w8_kernel(const int8_t* w, const half* scales, int k, int row_begin,
                                  half* out) {
  const int r = blockIdx.y;
  const size_t src_row = static_cast<size_t>(row_begin + r);
  const float s = __half2float(scales[src_row]);
  for (int c = (blockIdx.x * blockDim.x + threadIdx.x) * 8; c < k; c += gridDim.x * blockDim.x * 8) {
    const int8_t* src = w + src_row * k + c;
    float f[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = static_cast<float>(src[i]) * s;
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(r) * k + c) = pack_half8(f);
  }
}

__global__ void dequant_w4_kernel(const uint8_t* w, const half* scales, const half* offsets, int k,
                                  int group_size, int row_begin, half* out) {
  const int r = blockIdx.y;
  const size_t src_row = static_cast<size_t>(row_begin + r);
  const int groups = k / group_size;
  for (int c = (blockIdx.x * blockDim.x + threadIdx.x) * 8; c < k; c += gridDim.x * blockDim.x * 8) {
    const int g = c / group_size;
    const float s = __half2float(scales[src_row * groups + g]);
    const float o = __half2float(offsets[src_row * groups + g]);
    const uint8_t* src = w + src_row * (k / 2) + c / 2;
    float f[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[2 * i] = static_cast<float>(src[i] & 0xF) * s + o;
      f[2 * i + 1] = static_cast<float>(src[i] >> 4) * s + o;
    }
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(r) * k + c) = pack_half8(f);
  }
}

dim3 dequant_grid(int k, int rows) {
  const int vecs = k / 8;
  return dim3(std::max(1, std::min((vecs + 255) / 256, 8)), rows);
}

}  // namespace

void quantize_rows_int8(const half* x, int rows, int cols, int8_t* q, float* row_scales,
                        cudaStream_t stream) {
  if (rows <= 0) return;
  quantize_rows_int8_kernel<<<rows, 256, 0, stream>>>(x, cols, q, row_scales);
  RL_CUDA_CHECK_LAUNCH();
}

void dequantize_int32_gemm_output(const int32_t* acc, const float* row_scales,
                                  const half* col_scales, const half* bias, int rows, int cols,
                                  half* out, cudaStream_t stream) {
  if (rows <= 0) return;
  dequant_i32_kernel<<<dim3((cols + 255) / 256, rows), 256, 0, stream>>>(acc, row_scales, col_scales,
                                                                         bias, cols, out);
  RL_CUDA_CHECK_LAUNCH();
}

void scale_columns(half* y, const half* col_scales, float multiplier, const half* bias, int rows,
                   int cols, cudaStream_t stream) {
  if (rows <= 0) return;
  scale_columns_kernel<<<dim3((cols + 255) / 256, rows), 256, 0, stream>>>(y, col_scales, multiplier,
                                                                           bias, cols);
  RL_CUDA_CHECK_LAUNCH();
}

void dequantize_weight_int8(const int8_t* w, const half* scales, int k, int row_begin, int row_end,
                            half* out, cudaStream_t stream) {
  if (row_end <= row_begin) return;
  RL_CHECK(k % 8 == 0, "K must be a multiple of 8");
  dequant_w8_kernel<<<dequant_grid(k, row_end - row_begin), 256, 0, stream>>>(w, scales, k, row_begin,
                                                                              out);
  RL_CUDA_CHECK_LAUNCH();
}

void dequantize_weight_int4(const uint8_t* w, const half* scales, const half* offsets, int k,
                            int group_size, int row_begin, int row_end, half* out,
                            cudaStream_t stream) {
  if (row_end <= row_begin) return;
  RL_CHECK(k % group_size == 0 && group_size % 8 == 0, "bad INT4 shape");
  dequant_w4_kernel<<<dequant_grid(k, row_end - row_begin), 256, 0, stream>>>(
      w, scales, offsets, k, group_size, row_begin, out);
  RL_CUDA_CHECK_LAUNCH();
}

}  // namespace kernels
}  // namespace redline
