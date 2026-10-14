// Rotary position embedding fused with the paged KV-cache write.
#include "cuda/cuda_utils.h"
#include "cuda/kernels/common.cuh"
#include "cuda/kernels/kernels.h"

namespace redline {
namespace kernels {
namespace {

// One block per token, one warp per head. Lane l owns rotation pairs
// (i, i + D/2) for i = l + 32 j, so RoPE never crosses lanes. For INT8 KV
// each (token, kv head) gets its own absmax scale.
template <int D, typename KVT>
__global__ void rope_kv_kernel(half* qkv, const int32_t* positions, const int32_t* slot_mapping,
                               int num_heads, int num_kv_heads, const float* cos_table,
                               const float* sin_table, KVLayerView kv) {
  constexpr int kHalf = D / 2;
  constexpr int kPairs = D / 64;  // rotation pairs per lane
  const int t = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  const int pos = positions[t];
  const int slot = slot_mapping[t];
  const int total_heads = num_heads + 2 * num_kv_heads;
  half* row = qkv + static_cast<size_t>(t) * total_heads * D;

  float c[kPairs], s[kPairs];
#pragma unroll
  for (int j = 0; j < kPairs; ++j) {
    c[j] = cos_table[static_cast<size_t>(pos) * kHalf + lane + 32 * j];
    s[j] = sin_table[static_cast<size_t>(pos) * kHalf + lane + 32 * j];
  }

  for (int h = warp; h < total_heads; h += nwarps) {
    half* v = row + h * D;
    float x0[kPairs], x1[kPairs];
#pragma unroll
    for (int j = 0; j < kPairs; ++j) {
      x0[j] = __half2float(v[lane + 32 * j]);
      x1[j] = __half2float(v[lane + 32 * j + kHalf]);
    }
    const bool is_v = h >= num_heads + num_kv_heads;
    if (!is_v) {
#pragma unroll
      for (int j = 0; j < kPairs; ++j) {
        const float a = x0[j] * c[j] - x1[j] * s[j];
        const float b = x1[j] * c[j] + x0[j] * s[j];
        // Round like the fp16 activations HF would produce.
        x0[j] = __half2float(__float2half_rn(a));
        x1[j] = __half2float(__float2half_rn(b));
      }
    }
    if (h < num_heads) {  // query head: write back in place
#pragma unroll
      for (int j = 0; j < kPairs; ++j) {
        v[lane + 32 * j] = __float2half_rn(x0[j]);
        v[lane + 32 * j + kHalf] = __float2half_rn(x1[j]);
      }
      continue;
    }
    if (slot < 0) continue;
    const int kvh = is_v ? h - num_heads - num_kv_heads : h - num_heads;
    const int block = slot / kv.block_size, off = slot % kv.block_size;
    const size_t cell = (static_cast<size_t>(block) * num_kv_heads + kvh) * kv.block_size + off;
    KVT* dst = static_cast<KVT*>(is_v ? kv.v : kv.k) + cell * D;
    if constexpr (sizeof(KVT) == 2) {
#pragma unroll
      for (int j = 0; j < kPairs; ++j) {
        dst[lane + 32 * j] = __float2half_rn(x0[j]);
        dst[lane + 32 * j + kHalf] = __float2half_rn(x1[j]);
      }
    } else {
      float amax = 0.f;
#pragma unroll
      for (int j = 0; j < kPairs; ++j) amax = fmaxf(amax, fmaxf(fabsf(x0[j]), fabsf(x1[j])));
      amax = warp_reduce_max(amax);
      const float scale = amax > 0.f ? amax / 127.f : 1.f;
      const float inv = 1.f / scale;
#pragma unroll
      for (int j = 0; j < kPairs; ++j) {
        dst[lane + 32 * j] = static_cast<int8_t>(__float2int_rn(fminf(fmaxf(x0[j] * inv, -127.f), 127.f)));
        dst[lane + 32 * j + kHalf] =
            static_cast<int8_t>(__float2int_rn(fminf(fmaxf(x1[j] * inv, -127.f), 127.f)));
      }
      if (lane == 0) (is_v ? kv.v_scale : kv.k_scale)[cell] = scale;
    }
  }
}

template <int D>
void launch_rope_kv(half* qkv, const int32_t* positions, const int32_t* slots, int num_tokens,
                    int num_heads, const float* cos_t, const float* sin_t, const KVLayerView& kv,
                    cudaStream_t stream) {
  const int heads = num_heads + 2 * kv.num_kv_heads;
  const int threads = heads >= 8 ? 256 : 128;
  if (kv.dtype == KVDType::kFP16) {
    rope_kv_kernel<D, half><<<num_tokens, threads, 0, stream>>>(qkv, positions, slots, num_heads,
                                                                kv.num_kv_heads, cos_t, sin_t, kv);
  } else {
    rope_kv_kernel<D, int8_t><<<num_tokens, threads, 0, stream>>>(
        qkv, positions, slots, num_heads, kv.num_kv_heads, cos_t, sin_t, kv);
  }
  RL_CUDA_CHECK_LAUNCH();
}

}  // namespace

void rope_and_write_kv(half* qkv, const int32_t* positions, const int32_t* slot_mapping,
                       int num_tokens, int num_heads, const float* cos_table,
                       const float* sin_table, const KVLayerView& kv, cudaStream_t stream) {
  if (num_tokens <= 0) return;
  switch (kv.head_dim) {
    case 64:
      launch_rope_kv<64>(qkv, positions, slot_mapping, num_tokens, num_heads, cos_table, sin_table,
                         kv, stream);
      break;
    case 128:
      launch_rope_kv<128>(qkv, positions, slot_mapping, num_tokens, num_heads, cos_table,
                          sin_table, kv, stream);
      break;
    default:
      throw Error(str_cat("unsupported head_dim ", kv.head_dim, " (supported: 64, 128)"));
  }
}

}  // namespace kernels
}  // namespace redline
