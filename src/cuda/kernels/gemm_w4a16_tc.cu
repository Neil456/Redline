// W4A16 GEMM on tensor cores (mma.sync.m16n8k16, fp32 accumulate) for up to 64 rows: decode
// batches, speculative verification and short prefill chunks.
//
// Decode is memory-bound at batch 1, but past ~8 rows a CUDA-core GEMV becomes FMA- and
// L1-bound (each weight is multiplied by every row). Tensor cores do 16 rows x 8 columns x 16 k
// per instruction, so the kernel stays limited by weight bandwidth much longer.
//
// Layout and dequantization (see core/int4_layout.h): each lane loads 16 bytes per k-step that
// are exactly its B fragments for 8 n8 tiles (64 output columns per warp). Nibbles become fp16
// with the 0x6400 exponent trick (lop3 into the mantissa of 1024.0, then subtract), and the
// group scale / offset is applied with one fp16 FMA: w = q * s + o.
//
// Parallelism: a CTA has 4 warps over 256 adjacent output columns. K is split across a grid
// dimension so that enough CTAs stream weights concurrently; partial sums go to an fp32
// workspace and the last CTA of each column block (atomic ticket) adds them in a fixed order, so
// results are deterministic and the whole GEMM is a single launch (no separate reduce kernel).
// The split count depends only on (n, k), never on m, so per-row results are batch-invariant.
#include <cstdlib>

#include "cuda/cuda_utils.h"
#include "cuda/kernels/common.cuh"
#include "cuda/kernels/kernels.h"

namespace redline {
namespace kernels {
namespace {

constexpr int kTcWarps = 4;
constexpr int kTcThreads = 32 * kTcWarps;
constexpr int kTcCtaCols = 64 * kTcWarps;

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;  // (a & b) | c
  asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ half2 u32_as_h2(uint32_t v) {
  union {
    uint32_t u;
    half2 h;
  } x;
  x.u = v;
  return x.h;
}

__device__ __forceinline__ uint32_t h2_as_u32(half2 v) {
  union {
    half2 h;
    uint32_t u;
  } x;
  x.h = v;
  return x.u;
}

// 8 nibbles -> 4 half2: {b0,b1},{b2,b3} of n8 tile 2c, then of tile 2c+1 (values 0..15, exact).
__device__ __forceinline__ void dequant_word(uint32_t q, half2 (&f)[4]) {
  constexpr uint32_t kLo = 0x000f000fu, kHi = 0x00f000f0u, kExp = 0x64006400u;
  const half2 k1024 = u32_as_h2(0x64006400u);  // {1024, 1024}
  const half2 kInv16 = u32_as_h2(0x2c002c00u);  // {1/16, 1/16}
  const half2 kNeg64 = u32_as_h2(0xd400d400u);  // {-64, -64}
  f[0] = __hsub2(u32_as_h2(lop3_and_or(q, kLo, kExp)), k1024);
  f[1] = __hfma2(u32_as_h2(lop3_and_or(q, kHi, kExp)), kInv16, kNeg64);
  q >>= 8;
  f[2] = __hsub2(u32_as_h2(lop3_and_or(q, kLo, kExp)), k1024);
  f[3] = __hfma2(u32_as_h2(lop3_and_or(q, kHi, kExp)), kInv16, kNeg64);
}

__device__ __forceinline__ void mma_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ld_x2(const half* p) {
  return __ldg(reinterpret_cast<const uint32_t*>(p));
}

__device__ __forceinline__ float silu_mul(float g, float u) {
  // Same rounding points as the unfused path: fp16 GEMM outputs, fp16 SiLU, fp16 product.
  g = __half2float(__float2half_rn(g));
  u = __half2float(__float2half_rn(u));
  const float s = __half2float(__float2half_rn(g / (1.f + __expf(-g))));
  return s * u;
}

// MB: 16-row blocks (1..4). U: k-steps whose weights are loaded before any is used.
template <int MB, int U, bool kOutF32, bool kSiluMul>
__global__ void __launch_bounds__(kTcThreads)
gemm_w4a16_tc_kernel(const half* __restrict__ x, int m, const uint4* __restrict__ w,
                     const uint4* __restrict__ scales, const uint4* __restrict__ offsets,
                     int group_size, const half* __restrict__ bias, int n, int k, int k_per_split,
                     void* __restrict__ y, float* __restrict__ partials, int* __restrict__ tickets) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nt = blockIdx.x * kTcWarps + warp;  // 64-column tile of this warp
  const bool active = nt < n / 64;
  const int split = blockIdx.y, splits = gridDim.y;
  const int k0 = split * k_per_split, k1 = min(k, k0 + k_per_split);
  const int ksteps = k / 16, groups = k / group_size;

  float acc[MB][8][4];
#pragma unroll
  for (int mb = 0; mb < MB; ++mb)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mb][j][e] = 0.f;

  if (active) {
    const uint4* wt = w + static_cast<size_t>(nt) * ksteps * 32 + lane;
    const uint4* st = scales + static_cast<size_t>(nt) * groups * 8 + g;
    const uint4* ot = offsets + static_cast<size_t>(nt) * groups * 8 + g;
    int cur_group = -1;
    uint4 sv = make_uint4(0, 0, 0, 0), ov = sv;
    // Software pipeline: the next chunk's weights are in flight while this chunk computes, and a
    // chunk's activation fragments are loaded together (one L2 round trip per chunk, not per step).
    uint4 wv[U];
#pragma unroll
    for (int u = 0; u < U; ++u) wv[u] = __ldg(wt + static_cast<size_t>(k0 / 16 + u) * 32);
    for (int kk = k0; kk < k1; kk += 16 * U) {
      const bool has_next = kk + 16 * U < k1;
      uint4 wn[U];
      if (has_next) {
#pragma unroll
        for (int u = 0; u < U; ++u) wn[u] = __ldg(wt + static_cast<size_t>(kk / 16 + U + u) * 32);
      }
      uint32_t a[U][MB][4];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int kc = kk + 16 * u + 2 * t;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          const int r0 = mb * 16 + g, r1 = r0 + 8;
          const half* x0 = x + static_cast<size_t>(r0) * k + kc;
          const half* x1 = x + static_cast<size_t>(r1) * k + kc;
          a[u][mb][0] = r0 < m ? ld_x2(x0) : 0u;
          a[u][mb][1] = r1 < m ? ld_x2(x1) : 0u;
          a[u][mb][2] = r0 < m ? ld_x2(x0 + 8) : 0u;
          a[u][mb][3] = r1 < m ? ld_x2(x1 + 8) : 0u;
        }
      }
      const int grp = kk / group_size;  // U k-steps never cross a group (group_size % 16U == 0)
      if (grp != cur_group) {
        cur_group = grp;
        sv = __ldg(st + static_cast<size_t>(grp) * 8);
        ov = __ldg(ot + static_cast<size_t>(grp) * 8);
      }
      const half* sh = reinterpret_cast<const half*>(&sv);
      const half* oh = reinterpret_cast<const half*>(&ov);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t words[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          half2 f[4];
          dequant_word(words[c], f);
          const half2 s0 = __half2half2(sh[2 * c]), o0 = __half2half2(oh[2 * c]);
          const half2 s1 = __half2half2(sh[2 * c + 1]), o1 = __half2half2(oh[2 * c + 1]);
          const uint32_t b00 = h2_as_u32(__hfma2(f[0], s0, o0)), b01 = h2_as_u32(__hfma2(f[1], s0, o0));
          const uint32_t b10 = h2_as_u32(__hfma2(f[2], s1, o1)), b11 = h2_as_u32(__hfma2(f[3], s1, o1));
#pragma unroll
          for (int mb = 0; mb < MB; ++mb) {
            mma_16816(acc[mb][2 * c], a[u][mb], b00, b01);
            mma_16816(acc[mb][2 * c + 1], a[u][mb], b10, b11);
          }
        }
      }
      if (has_next) {
#pragma unroll
        for (int u = 0; u < U; ++u) wv[u] = wn[u];
      }
    }
  }

  if (splits > 1) {
    // Publish this CTA's partial tile, then let the last CTA of the column block reduce.
    float* part = partials + static_cast<size_t>(split) * m * n;
    if (active) {
#pragma unroll
      for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = mb * 16 + g + 8 * h;
            if (row < m) {
              const int col = nt * 64 + j * 8 + 2 * t;
              *reinterpret_cast<float2*>(part + static_cast<size_t>(row) * n + col) =
                  make_float2(acc[mb][j][2 * h], acc[mb][j][2 * h + 1]);
            }
          }
    }
    __threadfence();
    __syncthreads();
    __shared__ int is_last;
    if (threadIdx.x == 0) is_last = atomicAdd(&tickets[blockIdx.x], 1) == splits - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    if (active) {
      // Splits outermost: each split's loads for the whole tile are issued together, so the
      // reduction costs one L2 round trip per split rather than one per (output, split). The
      // per-element order is still s = 0, 1, ...: deterministic.
#pragma unroll
      for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[mb][j][0] = acc[mb][j][1] = acc[mb][j][2] = acc[mb][j][3] = 0.f;
#pragma unroll 2
      for (int s = 0; s < splits; ++s) {
        const float* ps = partials + static_cast<size_t>(s) * m * n;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int row = mb * 16 + g + 8 * h;
              if (row < m) {
                const float2 p = __ldcg(reinterpret_cast<const float2*>(
                    ps + static_cast<size_t>(row) * n + nt * 64 + j * 8 + 2 * t));
                acc[mb][j][2 * h] += p.x;
                acc[mb][j][2 * h + 1] += p.y;
              }
            }
      }
    }
    if (threadIdx.x == 0) tickets[blockIdx.x] = 0;  // ready for the next GEMM on this stream
  }

  if (!active) return;
#pragma unroll
  for (int mb = 0; mb < MB; ++mb)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = mb * 16 + g + 8 * h;
        if (row >= m) continue;
        const int col = nt * 64 + j * 8 + 2 * t;
        float v0 = acc[mb][j][2 * h], v1 = acc[mb][j][2 * h + 1];
        if constexpr (kSiluMul) {
          // Columns (2i, 2i + 1) are (gate_i, up_i): see int4_tc_source_row().
          static_cast<half*>(y)[static_cast<size_t>(row) * (n / 2) + col / 2] =
              __float2half_rn(silu_mul(v0, v1));
        } else {
          if (bias != nullptr) {
            v0 += __half2float(bias[col]);
            v1 += __half2float(bias[col + 1]);
          }
          if constexpr (kOutF32) {
            *reinterpret_cast<float2*>(static_cast<float*>(y) + static_cast<size_t>(row) * n + col) =
                make_float2(v0, v1);
          } else {
            *reinterpret_cast<half2*>(static_cast<half*>(y) + static_cast<size_t>(row) * n + col) =
                __floats2half2_rn(v0, v1);
          }
        }
      }
}

int num_sms() {
  static int sms = [] {
    int dev = 0, v = 0;
    RL_CUDA_CHECK(cudaGetDevice(&dev));
    RL_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
    return v;
  }();
  return sms;
}

template <int MB, int U>
void launch_mb(dim3 grid, const half* x, int m, const uint8_t* w, const half* scales,
               const half* offsets, int group_size, const half* bias, int n, int k, int kps,
               void* y, bool y_fp32, bool silu_mul, const W4TcWorkspace& ws, cudaStream_t s) {
  const auto* wp = reinterpret_cast<const uint4*>(w);
  const auto* sp = reinterpret_cast<const uint4*>(scales);
  const auto* op = reinterpret_cast<const uint4*>(offsets);
  if (silu_mul) {
    gemm_w4a16_tc_kernel<MB, U, false, true><<<grid, kTcThreads, 0, s>>>(
        x, m, wp, sp, op, group_size, bias, n, k, kps, y, ws.partials, ws.tickets);
  } else if (y_fp32) {
    gemm_w4a16_tc_kernel<MB, U, true, false><<<grid, kTcThreads, 0, s>>>(
        x, m, wp, sp, op, group_size, bias, n, k, kps, y, ws.partials, ws.tickets);
  } else {
    gemm_w4a16_tc_kernel<MB, U, false, false><<<grid, kTcThreads, 0, s>>>(
        x, m, wp, sp, op, group_size, bias, n, k, kps, y, ws.partials, ws.tickets);
  }
}

template <int MB>
void launch_u(dim3 grid, const half* x, int m, const uint8_t* w, const half* scales,
              const half* offsets, int group_size, const half* bias, int n, int k, int kps,
              void* y, bool y_fp32, bool silu_mul, const W4TcWorkspace& ws, cudaStream_t s) {
  // Prefetch depth: as many k-steps as fit in a group, capped to keep registers in check.
  constexpr int kMaxU = MB <= 2 ? 4 : 2;
  const int steps_per_group = group_size / 16;
  if (kMaxU >= 4 && steps_per_group % 4 == 0) {
    launch_mb<MB, (kMaxU >= 4 ? 4 : 2)>(grid, x, m, w, scales, offsets, group_size, bias, n, k, kps,
                                        y, y_fp32, silu_mul, ws, s);
  } else if (steps_per_group % 2 == 0) {
    launch_mb<MB, 2>(grid, x, m, w, scales, offsets, group_size, bias, n, k, kps, y, y_fp32,
                     silu_mul, ws, s);
  } else {
    launch_mb<MB, 1>(grid, x, m, w, scales, offsets, group_size, bias, n, k, kps, y, y_fp32,
                     silu_mul, ws, s);
  }
}

// Weight tc layout -> fp16 rows [row_begin, row_end) (multiples of 64) for the prefill path.
__global__ void dequant_w4_tc_kernel(const uint4* __restrict__ w, const uint4* __restrict__ scales,
                                     const uint4* __restrict__ offsets, int k, int group_size,
                                     int nt_begin, int num_tiles, half* __restrict__ out) {
  const int ksteps = k / 16, groups = k / group_size;
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if (warp_global >= num_tiles * ksteps) return;
  const int nt = nt_begin + warp_global / ksteps, ks = warp_global % ksteps;
  const int grp = ks * 16 / group_size;
  const uint4 wv = __ldg(w + (static_cast<size_t>(nt) * ksteps + ks) * 32 + lane);
  const uint4 sv = __ldg(scales + (static_cast<size_t>(nt) * groups + grp) * 8 + g);
  const uint4 ov = __ldg(offsets + (static_cast<size_t>(nt) * groups + grp) * 8 + g);
  const half* sh = reinterpret_cast<const half*>(&sv);
  const half* oh = reinterpret_cast<const half*>(&ov);
  const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
  for (int c = 0; c < 4; ++c) {
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int j = 2 * c + ((p >> 1) & 1);
      const int kk = ks * 16 + 2 * t + ((p >> 2) & 1) + 8 * (p & 1);
      const int row = (nt - nt_begin) * 64 + j * 8 + g;
      const float q = static_cast<float>((words[c] >> (4 * p)) & 0xF);
      out[static_cast<size_t>(row) * k + kk] =
          __float2half_rn(q * __half2float(sh[j]) + __half2float(oh[j]));
    }
  }
}

}  // namespace

// CTAs per SM the K split aims for (REDLINE_W4TC_CTAS_PER_SM overrides it for tuning).
int ctas_per_sm() {
  static int v = [] {
    const char* e = std::getenv("REDLINE_W4TC_CTAS_PER_SM");
    return e != nullptr && std::atoi(e) > 0 ? std::atoi(e) : 2;
  }();
  return v;
}

int w4a16_tc_splits(int n, int k, int group_size) {
  const int ctas = static_cast<int>(ceil_div(n, kTcCtaCols));
  const int groups = k / group_size;
  const int target = ctas_per_sm() * num_sms();
  int splits = std::max(1, std::min(groups, static_cast<int>(ceil_div(target, ctas))));
  const int groups_per_split = static_cast<int>(ceil_div(groups, splits));
  return static_cast<int>(ceil_div(groups, groups_per_split));
}

size_t w4a16_tc_workspace_floats(int max_m, int n, int k, int group_size) {
  const int splits = w4a16_tc_splits(n, k, group_size);
  return splits > 1 ? static_cast<size_t>(splits) * max_m * n : 0;
}

void gemm_w4a16_tc(const half* x, int m, const uint8_t* w, const half* scales, const half* offsets,
                   int group_size, const half* bias, int n, int k, void* y, bool y_fp32,
                   bool silu_mul, const W4TcWorkspace& ws, cudaStream_t stream) {
  if (m <= 0) return;
  RL_CHECK(m <= kW4TcMaxM, "gemm_w4a16_tc: m=", m, " exceeds ", kW4TcMaxM);
  RL_CHECK(n % 64 == 0 && k % 16 == 0 && group_size % 16 == 0 && k % group_size == 0,
           "gemm_w4a16_tc: unsupported shape n=", n, " k=", k, " g=", group_size);
  RL_CHECK(!silu_mul || (!y_fp32 && bias == nullptr), "fused SiLU epilogue is fp16 without bias");
  const int splits = w4a16_tc_splits(n, k, group_size);
  const int groups = k / group_size;
  const int kps = static_cast<int>(ceil_div(groups, splits)) * group_size;
  const dim3 grid(static_cast<unsigned>(ceil_div(n, kTcCtaCols)), static_cast<unsigned>(splits));
  if (splits > 1) {
    RL_CHECK(static_cast<size_t>(splits) * m * n <= ws.partial_floats, "W4 tensor-core workspace too small (",
             splits, " splits x ", m, " x ", n, ")");
    RL_CHECK(static_cast<int>(grid.x) <= ws.num_tickets, "W4 tensor-core ticket array too small");
  }
  const int mb = static_cast<int>(ceil_div(m, 16));
  switch (mb) {
    case 1: launch_u<1>(grid, x, m, w, scales, offsets, group_size, bias, n, k, kps, y, y_fp32, silu_mul, ws, stream); break;
    case 2: launch_u<2>(grid, x, m, w, scales, offsets, group_size, bias, n, k, kps, y, y_fp32, silu_mul, ws, stream); break;
    case 3: launch_u<3>(grid, x, m, w, scales, offsets, group_size, bias, n, k, kps, y, y_fp32, silu_mul, ws, stream); break;
    default: launch_u<4>(grid, x, m, w, scales, offsets, group_size, bias, n, k, kps, y, y_fp32, silu_mul, ws, stream); break;
  }
  RL_CUDA_CHECK_LAUNCH();
}

void dequantize_weight_int4_tc(const uint8_t* w, const half* scales, const half* offsets, int k,
                               int group_size, int row_begin, int row_end, half* out,
                               cudaStream_t stream) {
  if (row_end <= row_begin) return;
  RL_CHECK(row_begin % 64 == 0 && row_end % 64 == 0, "tc dequant needs 64-row aligned ranges");
  const int tiles = (row_end - row_begin) / 64;
  const int warps = tiles * (k / 16);
  const int threads = 256;
  const int blocks = static_cast<int>(ceil_div(static_cast<int64_t>(warps) * 32, threads));
  dequant_w4_tc_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(w), reinterpret_cast<const uint4*>(scales),
      reinterpret_cast<const uint4*>(offsets), k, group_size, row_begin / 64, tiles, out);
  RL_CUDA_CHECK_LAUNCH();
}

}  // namespace kernels
}  // namespace redline
