// Device-side helpers shared by the kernels.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace redline {
namespace kernels {

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFullMask, v, o);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFullMask, v, o));
  return v;
}

// Block-wide reductions; `scratch` needs blockDim.x / 32 floats. All threads
// receive the result. Contains __syncthreads, so call from uniform control flow.
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_reduce_sum(v);
  __syncthreads();  // protect scratch from a previous use
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = lane < nwarps ? scratch[lane] : 0.f;
  return warp_reduce_sum(v);
}

__device__ __forceinline__ float block_reduce_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_reduce_max(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = lane < nwarps ? scratch[lane] : -FLT_MAX;
  return warp_reduce_max(v);
}

// (value, index) argmax; ties resolve to the smaller index.
struct ArgMax {
  float v;
  int i;
};

__device__ __forceinline__ ArgMax argmax_combine(ArgMax a, ArgMax b) {
  if (b.v > a.v || (b.v == a.v && b.i < a.i)) return b;
  return a;
}

__device__ __forceinline__ ArgMax warp_reduce_argmax(ArgMax a) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ArgMax b;
    b.v = __shfl_xor_sync(kFullMask, a.v, o);
    b.i = __shfl_xor_sync(kFullMask, a.i, o);
    a = argmax_combine(a, b);
  }
  return a;
}

// `scratch` needs blockDim.x / 32 entries.
__device__ __forceinline__ ArgMax block_reduce_argmax(ArgMax a, ArgMax* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  a = warp_reduce_argmax(a);
  __syncthreads();
  if (lane == 0) scratch[warp] = a;
  __syncthreads();
  ArgMax r = lane < nwarps ? scratch[lane] : ArgMax{-FLT_MAX, 0x7fffffff};
  return warp_reduce_argmax(r);
}

// 8 halves <-> float[8] through a 16-byte vector.
__device__ __forceinline__ void unpack_half8(const uint4& v, float* f) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack_half8(const float* f) {
  uint4 v;
  __half2* h = reinterpret_cast<__half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
  return v;
}

// 16 int8 values from a 16-byte vector.
__device__ __forceinline__ void unpack_int8x16(const uint4& v, float* f) {
  const int8_t* b = reinterpret_cast<const int8_t*>(&v);
#pragma unroll
  for (int i = 0; i < 16; ++i) f[i] = static_cast<float>(b[i]);
}

// Order-preserving map from float to uint32 (larger float -> larger key).
__device__ __forceinline__ uint32_t float_to_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Philox4x32-10 counter-based RNG (Salmon et al., 2011).
__device__ __forceinline__ uint4 philox4x32_10(uint4 ctr, uint2 key) {
  constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u;
  constexpr uint32_t kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    const uint32_t hi0 = __umulhi(kM0, ctr.x), lo0 = kM0 * ctr.x;
    const uint32_t hi1 = __umulhi(kM1, ctr.z), lo1 = kM1 * ctr.z;
    ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
    key.x += kW0;
    key.y += kW1;
  }
  return ctr;
}

// Uniform float in the open interval (0, 1) for (seed, position, stream, index).
__device__ __forceinline__ float uniform01(uint32_t seed_lo, uint32_t seed_hi, uint32_t position,
                                           uint32_t stream, uint32_t index) {
  const uint4 r = philox4x32_10(make_uint4(index, position, stream, 0x52444C4Eu),
                                make_uint2(seed_lo, seed_hi));
  return (static_cast<float>(r.x >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float gumbel(float u) { return -__logf(-__logf(u)); }

}  // namespace kernels
}  // namespace redline
