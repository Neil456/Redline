// Weight-only quantized GEMV kernels for decode (M <= 8 rows for INT8, M <= 16 for INT4).
//
// Decode is memory-bound: every step streams all weights once. These kernels
// read weights with 16-byte loads, dequantize in registers and accumulate in
// fp32, so INT8 halves and INT4 quarters the weight traffic of FP16.
// Each warp computes kRows output channels for all M rows; lanes split K.
#include "cuda/cuda_utils.h"
#include "cuda/kernels/common.cuh"
#include "cuda/kernels/kernels.h"

namespace redline {
namespace kernels {
namespace {

constexpr int kGemvThreads = 128;
constexpr int kGemvWarps = kGemvThreads / 32;
constexpr int kRows = 2;  // output channels per warp

template <int M, bool kOutF32>
__device__ __forceinline__ void store_outputs(float (&acc)[kRows][M], int row0, int n,
                                              const half* scales, const half* bias, void* y,
                                              int lane, bool per_channel_scale) {
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = warp_reduce_sum(acc[r][m]);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int row = row0 + r;
    if (row >= n) continue;
    const float s = per_channel_scale ? __half2float(scales[row]) : 1.f;
    const float b = bias ? __half2float(bias[row]) : 0.f;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float v = acc[r][m] * s + b;
      if constexpr (kOutF32) {
        static_cast<float*>(y)[static_cast<size_t>(m) * n + row] = v;
      } else {
        static_cast<half*>(y)[static_cast<size_t>(m) * n + row] = __float2half_rn(v);
      }
    }
  }
}

// Rows (row0, row0 + 1) hold (gate_i, up_i) with i = row0 / 2: writes silu(gate) * up, fp16
// [m, n / 2], with the rounding points of the unfused path (fp16 GEMM outputs, fp16 SiLU).
template <int M>
__device__ __forceinline__ void store_swiglu(float (&acc)[kRows][M], int row0, int n, void* y, int lane) {
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = warp_reduce_sum(acc[r][m]);
  if (lane != 0 || row0 + 1 >= n) return;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float g = __half2float(__float2half_rn(acc[0][m]));
    const float u = __half2float(__float2half_rn(acc[1][m]));
    const float s = __half2float(__float2half_rn(g / (1.f + __expf(-g))));
    static_cast<half*>(y)[static_cast<size_t>(m) * (n / 2) + row0 / 2] = __float2half_rn(s * u);
  }
}

// INT8 per-channel: each lane consumes 16 consecutive k per iteration.
template <int M, bool kOutF32>
__global__ void __launch_bounds__(kGemvThreads)
gemv_w8a16_kernel(const half* __restrict__ x, const int8_t* __restrict__ w,
                  const half* __restrict__ scales, const half* __restrict__ bias, int n, int k,
                  void* __restrict__ y) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kGemvWarps + (threadIdx.x >> 5)) * kRows;
  if (row0 >= n) return;
  float acc[kRows][M];
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  for (int kk = lane * 16; kk < k; kk += 32 * 16) {
    float wf[kRows][16];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int row = min(row0 + r, n - 1);  // duplicate the last row; never stored
      unpack_int8x16(__ldg(reinterpret_cast<const uint4*>(w + static_cast<size_t>(row) * k + kk)), wf[r]);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * k + kk);
      float xf[16];
      unpack_half8(__ldg(xp), xf);
      unpack_half8(__ldg(xp + 1), xf + 8);
#pragma unroll
      for (int r = 0; r < kRows; ++r)
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[r][m] += wf[r][e] * xf[e];
    }
  }
  store_outputs<M, kOutF32>(acc, row0, n, scales, bias, y, lane, true);
}

// INT4 group-wise: each lane consumes 32 consecutive k (16 bytes) per
// iteration; the 32 values share one (scale, offset) because group_size is a
// multiple of 32. y = s * sum(x * q) + o * sum(x).
// kMode: 0 = fp16 out, 1 = fp32 out, 2 = SwiGLU of interleaved (gate, up) rows.
template <int M, int kMode>
__global__ void __launch_bounds__(kGemvThreads)
gemv_w4a16_kernel(const half* __restrict__ x, const uint8_t* __restrict__ w,
                  const half* __restrict__ scales, const half* __restrict__ offsets,
                  int group_size, const half* __restrict__ bias, int n, int k,
                  void* __restrict__ y) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kGemvWarps + (threadIdx.x >> 5)) * kRows;
  if (row0 >= n) return;
  const int groups = k / group_size;
  float acc[kRows][M];
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  for (int kk = lane * 32; kk < k; kk += 32 * 32) {
    const int g = kk / group_size;
    uint4 packed[kRows];
    float s[kRows], o[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int row = min(row0 + r, n - 1);
      packed[r] = __ldg(reinterpret_cast<const uint4*>(w + static_cast<size_t>(row) * (k / 2) + kk / 2));
      s[r] = __half2float(scales[static_cast<size_t>(row) * groups + g]);
      o[r] = __half2float(offsets[static_cast<size_t>(row) * groups + g]);
    }
    float dot[kRows][M], xsum[M];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      xsum[m] = 0.f;
#pragma unroll
      for (int r = 0; r < kRows; ++r) dot[r][m] = 0.f;
    }
    // Unpack each 8-weight word once and reuse it for every activation row, so the
    // ALU cost per weight does not grow with M (it dominates for M > 8).
#pragma unroll
    for (int v = 0; v < 4; ++v) {  // 4 x 8 nibbles = 32 values
      float wf[kRows][8];
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint32_t word = reinterpret_cast<const uint32_t*>(&packed[r])[v];
#pragma unroll
        for (int e = 0; e < 8; ++e) wf[r][e] = static_cast<float>((word >> (4 * e)) & 0xF);
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[8];
        unpack_half8(__ldg(reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * k + kk) + v), xf);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          xsum[m] += xf[e];
#pragma unroll
          for (int r = 0; r < kRows; ++r) dot[r][m] += xf[e] * wf[r][e];
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[r][m] += s[r] * dot[r][m] + o[r] * xsum[m];
  }
  if constexpr (kMode == 2) {
    store_swiglu<M>(acc, row0, n, y, lane);
  } else {
    store_outputs<M, kMode == 1>(acc, row0, n, nullptr, bias, y, lane, false);
  }
}

int gemv_blocks(int n) { return (n + kGemvWarps * kRows - 1) / (kGemvWarps * kRows); }

template <bool kOutF32>
void launch_w8(const half* x, int m, const int8_t* w, const half* scales, const half* bias, int n,
               int k, void* y, cudaStream_t s) {
  const int blocks = gemv_blocks(n);
#define RL_W8_CASE(MM)                                                                  \
  case MM:                                                                              \
    gemv_w8a16_kernel<MM, kOutF32><<<blocks, kGemvThreads, 0, s>>>(x, w, scales, bias, n, k, y); \
    break;
  switch (m) {
    RL_W8_CASE(1) RL_W8_CASE(2) RL_W8_CASE(3) RL_W8_CASE(4)
    RL_W8_CASE(5) RL_W8_CASE(6) RL_W8_CASE(7) RL_W8_CASE(8)
    RL_W8_CASE(9) RL_W8_CASE(10) RL_W8_CASE(11) RL_W8_CASE(12)
    RL_W8_CASE(13) RL_W8_CASE(14) RL_W8_CASE(15) RL_W8_CASE(16)
    default: throw Error(str_cat("gemv_w8a16: unsupported m=", m));
  }
#undef RL_W8_CASE
  RL_CUDA_CHECK_LAUNCH();
}

template <int kMode>
void launch_w4(const half* x, int m, const uint8_t* w, const half* scales, const half* offsets,
               int group_size, const half* bias, int n, int k, void* y, cudaStream_t s) {
  const int blocks = gemv_blocks(n);
#define RL_W4_CASE(MM)                                                                     \
  case MM:                                                                                 \
    gemv_w4a16_kernel<MM, kMode><<<blocks, kGemvThreads, 0, s>>>(x, w, scales, offsets,    \
                                                                 group_size, bias, n, k, y); \
    break;
  switch (m) {
    RL_W4_CASE(1) RL_W4_CASE(2) RL_W4_CASE(3) RL_W4_CASE(4)
    RL_W4_CASE(5) RL_W4_CASE(6) RL_W4_CASE(7) RL_W4_CASE(8)
    RL_W4_CASE(9) RL_W4_CASE(10) RL_W4_CASE(11) RL_W4_CASE(12)
    RL_W4_CASE(13) RL_W4_CASE(14) RL_W4_CASE(15) RL_W4_CASE(16)
    default: throw Error(str_cat("gemv_w4a16: unsupported m=", m));
  }
#undef RL_W4_CASE
  RL_CUDA_CHECK_LAUNCH();
}

}  // namespace

void gemv_w8a16(const half* x, int m, const int8_t* w, const half* scales, const half* bias, int n,
                int k, void* y, bool y_fp32, cudaStream_t stream) {
  if (m <= 0) return;
  RL_CHECK(k % 16 == 0, "gemv_w8a16 needs K % 16 == 0, got ", k);
  if (y_fp32) {
    launch_w8<true>(x, m, w, scales, bias, n, k, y, stream);
  } else {
    launch_w8<false>(x, m, w, scales, bias, n, k, y, stream);
  }
}

void gemv_w4a16(const half* x, int m, const uint8_t* w, const half* scales, const half* offsets,
                int group_size, const half* bias, int n, int k, void* y, bool y_fp32,
                cudaStream_t stream) {
  if (m <= 0) return;
  RL_CHECK(k % 32 == 0 && group_size % 32 == 0 && k % group_size == 0,
           "gemv_w4a16 needs K and group_size multiples of 32");
  if (y_fp32) {
    launch_w4<1>(x, m, w, scales, offsets, group_size, bias, n, k, y, stream);
  } else {
    launch_w4<0>(x, m, w, scales, offsets, group_size, bias, n, k, y, stream);
  }
}

void gemv_w4a16_swiglu(const half* x, int m, const uint8_t* w, const half* scales,
                       const half* offsets, int group_size, int n, int k, half* y,
                       cudaStream_t stream) {
  if (m <= 0) return;
  RL_CHECK(k % 32 == 0 && group_size % 32 == 0 && k % group_size == 0 && n % 2 == 0,
           "gemv_w4a16_swiglu: bad shape");
  launch_w4<2>(x, m, w, scales, offsets, group_size, nullptr, n, k, y, stream);
}

}  // namespace kernels
}  // namespace redline
