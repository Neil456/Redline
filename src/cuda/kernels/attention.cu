// Paged attention over the block-structured KV cache.
//
// Decode: flash-decoding style split-K. A CTA handles one (query token,
// kv head, key partition) and all G query heads that share that kv head, so
// every K/V vector is loaded once per GQA group instead of once per query
// head. Partials are merged by a small reduction kernel when a context spans
// more than one partition.
//
// Prefill: flash-attention style tiling with an online softmax. A CTA handles
// a 32-query tile of one head and streams 32-key tiles of K and V through
// shared memory, so memory traffic is linear in the context length.
#include <cfloat>

#include "cuda/cuda_utils.h"
#include "cuda/kernels/common.cuh"
#include "cuda/kernels/kernels.h"

namespace redline {
namespace kernels {
namespace {

constexpr int kDecodeThreads = 128;
constexpr int kDecodeWarps = kDecodeThreads / 32;
constexpr int kPS = kDecodePartitionSize;

// Loads 16 bytes of a cached K/V row starting at element `e` as floats.
// Returns the number of elements produced (8 for fp16, 16 for int8).
template <typename KVT>
struct KVChunk;

template <>
struct KVChunk<half> {
  static constexpr int kElems = 8;
  __device__ static void load(const half* row, int e, float* out) {
    unpack_half8(*reinterpret_cast<const uint4*>(row + e), out);
  }
};

template <>
struct KVChunk<int8_t> {
  static constexpr int kElems = 16;
  __device__ static void load(const int8_t* row, int e, float* out) {
    unpack_int8x16(*reinterpret_cast<const uint4*>(row + e), out);
  }
};

// Loads D/32 consecutive elements for lane `lane` of a V row.
template <int D, typename KVT>
__device__ __forceinline__ void load_v_lane(const KVT* row, int lane, float* out) {
  constexpr int kPer = D / 32;
  if constexpr (sizeof(KVT) == 2) {
    const half* p = reinterpret_cast<const half*>(row) + lane * kPer;
    if constexpr (kPer == 4) {
      const uint2 v = *reinterpret_cast<const uint2*>(p);
      const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
      const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
      out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
    } else {
      const float2 a = __half22float2(*reinterpret_cast<const __half2*>(p));
      out[0] = a.x; out[1] = a.y;
    }
  } else {
    const int8_t* p = reinterpret_cast<const int8_t*>(row) + lane * kPer;
#pragma unroll
    for (int i = 0; i < kPer; ++i) out[i] = static_cast<float>(p[i]);
  }
}

template <int D, int G, typename KVT>
__global__ void __launch_bounds__(kDecodeThreads, 4)
decode_attention_kernel(AttentionParams p, DecodeAttentionWork w) {
  __shared__ float q_s[G][D];
  __shared__ float scores[G][kPS];
  __shared__ float red[kDecodeWarps][G][D];
  __shared__ float scratch[kDecodeWarps];
  __shared__ float row_max[G], row_sum[G];

  const int part = blockIdx.x, kvh = blockIdx.y, item = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int token = w.item_token[item];
  const int seq = w.item_seq[item];
  const int ctx = p.positions[token] + 1;
  const int num_parts = (ctx + kPS - 1) / kPS;
  if (part >= num_parts) return;
  const int start = part * kPS;
  const int n = min(kPS, ctx - start);
  const int bs = p.kv.block_size, hkv = p.kv.num_kv_heads;
  const int32_t* table = p.block_tables + static_cast<size_t>(seq) * p.max_blocks_per_seq;
  const KVT* k_cache = static_cast<const KVT*>(p.kv.k);
  const KVT* v_cache = static_cast<const KVT*>(p.kv.v);

  for (int i = tid; i < G * D; i += kDecodeThreads) {
    const int g = i / D, d = i % D;
    const int h = kvh * G + g;
    q_s[g][d] = __half2float(p.q[static_cast<size_t>(token) * p.q_stride + h * D + d]) * p.scale;
  }
  __syncthreads();

  // Phase 1: scores. One thread per key; q is broadcast from shared memory.
  for (int j = tid; j < n; j += kDecodeThreads) {
    const int pos = start + j;
    const int block = table[pos / bs];
    const size_t cell = (static_cast<size_t>(block) * hkv + kvh) * bs + pos % bs;
    const KVT* krow = k_cache + cell * D;
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) acc[g] = 0.f;
#pragma unroll 1
    for (int e = 0; e < D; e += KVChunk<KVT>::kElems) {
      float kf[KVChunk<KVT>::kElems];
      KVChunk<KVT>::load(krow, e, kf);
#pragma unroll
      for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int x = 0; x < KVChunk<KVT>::kElems; ++x) acc[g] += q_s[g][e + x] * kf[x];
      }
    }
    const float ks = sizeof(KVT) == 1 ? p.kv.k_scale[cell] : 1.f;
#pragma unroll
    for (int g = 0; g < G; ++g) scores[g][j] = acc[g] * ks;
  }
  __syncthreads();

  // Softmax statistics per query head.
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float m = -FLT_MAX;
    for (int j = tid; j < n; j += kDecodeThreads) m = fmaxf(m, scores[g][j]);
    m = block_reduce_max(m, scratch);
    float s = 0.f;
    for (int j = tid; j < n; j += kDecodeThreads) {
      const float e = __expf(scores[g][j] - m);
      scores[g][j] = e;
      s += e;
    }
    s = block_reduce_sum(s, scratch);
    if (tid == 0) {
      row_max[g] = m;
      row_sum[g] = s;
    }
  }
  __syncthreads();

  // Phase 2: P @ V. Warp w takes keys j = w mod 4; lane owns D/32 dims.
  constexpr int kPer = D / 32;
  float acc[G][kPer];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int x = 0; x < kPer; ++x) acc[g][x] = 0.f;
  for (int j = warp; j < n; j += kDecodeWarps) {
    const int pos = start + j;
    const int block = table[pos / bs];
    const size_t cell = (static_cast<size_t>(block) * hkv + kvh) * bs + pos % bs;
    float vf[kPer];
    load_v_lane<D, KVT>(v_cache + cell * D, lane, vf);
    const float vs = sizeof(KVT) == 1 ? p.kv.v_scale[cell] : 1.f;
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float pj = scores[g][j] * vs;
#pragma unroll
      for (int x = 0; x < kPer; ++x) acc[g][x] += pj * vf[x];
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int x = 0; x < kPer; ++x) red[warp][g][lane * kPer + x] = acc[g][x];
  __syncthreads();

  for (int i = tid; i < G * D; i += kDecodeThreads) {
    const int g = i / D, d = i % D;
    float o = 0.f;
#pragma unroll
    for (int ww = 0; ww < kDecodeWarps; ++ww) o += red[ww][g][d];
    o /= row_sum[g];
    const int h = kvh * G + g;
    if (num_parts == 1) {
      p.out[static_cast<size_t>(token) * p.num_heads * D + h * D + d] = __float2half_rn(o);
    } else {
      const size_t base = (static_cast<size_t>(item) * p.num_heads + h) * w.max_partitions + part;
      w.ws_out[base * D + d] = o;
      if (d == 0) {
        w.ws_max[base] = row_max[g];
        w.ws_sum[base] = row_sum[g];
      }
    }
  }
}

// Merges split-K partials: one CTA per (item, head), one thread per dim.
template <int D>
__global__ void decode_reduce_kernel(AttentionParams p, DecodeAttentionWork w) {
  const int item = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int token = w.item_token[item];
  const int ctx = p.positions[token] + 1;
  const int num_parts = (ctx + kPS - 1) / kPS;
  if (num_parts <= 1) return;
  const size_t base = (static_cast<size_t>(item) * p.num_heads + h) * w.max_partitions;
  float m = -FLT_MAX;
  for (int i = 0; i < num_parts; ++i) m = fmaxf(m, w.ws_max[base + i]);
  float total = 0.f, o = 0.f;
  for (int i = 0; i < num_parts; ++i) {
    const float wgt = __expf(w.ws_max[base + i] - m) * w.ws_sum[base + i];
    total += wgt;
    o += wgt * w.ws_out[(base + i) * D + d];
  }
  p.out[static_cast<size_t>(token) * p.num_heads * D + h * D + d] = __float2half_rn(o / total);
}

// ---------------------------------------------------------------- prefill

constexpr int kBQ = kPrefillTileQ;
constexpr int kBK = 32;
constexpr int kPrefillThreads = 128;  // 4 threads per query row

template <int D, typename KVT>
__global__ void __launch_bounds__(kPrefillThreads)
prefill_attention_kernel(AttentionParams p, PrefillAttentionWork w) {
  constexpr int kPad = 8;  // keeps rows 16B aligned and spreads banks
  constexpr int kLd = D + kPad;
  __shared__ __align__(16) half q_s[kBQ][kLd];
  __shared__ __align__(16) half k_s[kBK][kLd];
  __shared__ __align__(16) half v_s[kBK][kLd];
  __shared__ float s_s[kBQ][kBK + 1];

  const int qt = blockIdx.x, h = blockIdx.y;
  const int seq = w.seq_ids[blockIdx.z];
  const int q_len = w.q_len[seq];
  if (qt * kBQ >= q_len) return;
  const int q0 = w.q_start[seq];
  const int ctx = w.ctx_len[seq];
  const int past = ctx - q_len;  // tokens already in the cache before this step
  const int G = p.num_heads / p.kv.num_kv_heads;
  const int kvh = h / G;
  const int bs = p.kv.block_size, hkv = p.kv.num_kv_heads;
  const int32_t* table = p.block_tables + static_cast<size_t>(seq) * p.max_blocks_per_seq;
  const int tid = threadIdx.x;
  const int r = tid >> 2;     // query row owned by this thread
  const int c0 = tid & 3;     // column phase

  // Load the query tile.
  constexpr int kChunksPerRow = D / 8;
  for (int i = tid; i < kBQ * kChunksPerRow; i += kPrefillThreads) {
    const int row = i / kChunksPerRow, ch = i % kChunksPerRow;
    const int qi = qt * kBQ + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (qi < q_len) {
      v = *reinterpret_cast<const uint4*>(p.q + static_cast<size_t>(q0 + qi) * p.q_stride + h * D +
                                          ch * 8);
    }
    *reinterpret_cast<uint4*>(&q_s[row][ch * 8]) = v;
  }

  const int my_q = qt * kBQ + r;            // index among the new tokens
  const int my_pos = past + my_q;           // absolute position
  const int last_q = min(q_len, (qt + 1) * kBQ) - 1;
  const int kv_end = past + last_q + 1;     // keys needed by this tile
  constexpr int kDPer = D / 4;              // output dims per thread
  float o[kDPer];
#pragma unroll
  for (int i = 0; i < kDPer; ++i) o[i] = 0.f;
  float m_run = -FLT_MAX, l_run = 0.f;

  for (int k0 = 0; k0 < kv_end; k0 += kBK) {
    __syncthreads();  // previous tile fully consumed (and q tile visible on first pass)
    // Load K and V tiles, dequantizing INT8 into half.
    for (int i = tid; i < kBK * kChunksPerRow; i += kPrefillThreads) {
      const int row = i / kChunksPerRow, ch = i % kChunksPerRow;
      const int pos = k0 + row;
      uint4 kv4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
      if (pos < kv_end) {
        const int block = table[pos / bs];
        const size_t cell = (static_cast<size_t>(block) * hkv + kvh) * bs + pos % bs;
        if constexpr (sizeof(KVT) == 2) {
          kv4 = *reinterpret_cast<const uint4*>(static_cast<const half*>(p.kv.k) + cell * D + ch * 8);
          vv4 = *reinterpret_cast<const uint4*>(static_cast<const half*>(p.kv.v) + cell * D + ch * 8);
        } else {
          const int8_t* kp = static_cast<const int8_t*>(p.kv.k) + cell * D + ch * 8;
          const int8_t* vp = static_cast<const int8_t*>(p.kv.v) + cell * D + ch * 8;
          const float ks = p.kv.k_scale[cell], vs = p.kv.v_scale[cell];
          float kf[8], vf[8];
#pragma unroll
          for (int x = 0; x < 8; ++x) {
            kf[x] = static_cast<float>(kp[x]) * ks;
            vf[x] = static_cast<float>(vp[x]) * vs;
          }
          kv4 = pack_half8(kf);
          vv4 = pack_half8(vf);
        }
      }
      *reinterpret_cast<uint4*>(&k_s[row][ch * 8]) = kv4;
      *reinterpret_cast<uint4*>(&v_s[row][ch * 8]) = vv4;
    }
    __syncthreads();

    // S = Q K^T for row r, columns c0 + 4 jj.
    float s[kBK / 4];
#pragma unroll
    for (int jj = 0; jj < kBK / 4; ++jj) s[jj] = 0.f;
#pragma unroll 4
    for (int d = 0; d < D; d += 2) {
      const float2 qf = __half22float2(*reinterpret_cast<const __half2*>(&q_s[r][d]));
#pragma unroll
      for (int jj = 0; jj < kBK / 4; ++jj) {
        const float2 kf = __half22float2(*reinterpret_cast<const __half2*>(&k_s[c0 + 4 * jj][d]));
        s[jj] += qf.x * kf.x + qf.y * kf.y;
      }
    }
    float tile_max = -FLT_MAX;
#pragma unroll
    for (int jj = 0; jj < kBK / 4; ++jj) {
      const int kpos = k0 + c0 + 4 * jj;
      const bool valid = my_q < q_len && kpos <= my_pos;
      s[jj] = valid ? s[jj] * p.scale : -FLT_MAX;
      tile_max = fmaxf(tile_max, s[jj]);
    }
    // The 4 threads of a row are adjacent lanes.
    tile_max = fmaxf(tile_max, __shfl_xor_sync(kFullMask, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(kFullMask, tile_max, 2));
    const float m_new = fmaxf(m_run, tile_max);
    const float alpha = m_run == -FLT_MAX ? 0.f : __expf(m_run - m_new);
    float tile_sum = 0.f;
#pragma unroll
    for (int jj = 0; jj < kBK / 4; ++jj) {
      const float e = s[jj] == -FLT_MAX ? 0.f : __expf(s[jj] - m_new);
      s_s[r][c0 + 4 * jj] = e;
      tile_sum += e;
    }
    tile_sum += __shfl_xor_sync(kFullMask, tile_sum, 1);
    tile_sum += __shfl_xor_sync(kFullMask, tile_sum, 2);
    l_run = l_run * alpha + tile_sum;
    m_run = m_new;
#pragma unroll
    for (int i = 0; i < kDPer; ++i) o[i] *= alpha;
    __syncwarp();

    // O += P V: thread owns dims c0*2 + 8 i + {0, 1}.
    for (int j = 0; j < kBK; ++j) {
      const float pj = s_s[r][j];
#pragma unroll
      for (int i = 0; i < kDPer / 2; ++i) {
        const float2 vf = __half22float2(*reinterpret_cast<const __half2*>(&v_s[j][c0 * 2 + 8 * i]));
        o[2 * i] += pj * vf.x;
        o[2 * i + 1] += pj * vf.y;
      }
    }
  }

  if (my_q < q_len) {
    const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
    half* out = p.out + static_cast<size_t>(q0 + my_q) * p.num_heads * D + h * D;
#pragma unroll
    for (int i = 0; i < kDPer / 2; ++i) {
      *reinterpret_cast<__half2*>(out + c0 * 2 + 8 * i) =
          __floats2half2_rn(o[2 * i] * inv, o[2 * i + 1] * inv);
    }
  }
}

template <int D, int G, typename KVT>
void launch_decode(const AttentionParams& p, const DecodeAttentionWork& w, cudaStream_t stream) {
  dim3 grid(w.max_partitions, p.kv.num_kv_heads, w.num_items);
  decode_attention_kernel<D, G, KVT><<<grid, kDecodeThreads, 0, stream>>>(p, w);
  RL_CUDA_CHECK_LAUNCH();
  if (w.max_partitions > 1) {
    decode_reduce_kernel<D><<<dim3(w.num_items, p.num_heads), D, 0, stream>>>(p, w);
    RL_CUDA_CHECK_LAUNCH();
  }
}

template <int D, typename KVT>
void dispatch_decode_group(const AttentionParams& p, const DecodeAttentionWork& w,
                           cudaStream_t stream) {
  switch (p.num_heads / p.kv.num_kv_heads) {
    case 1: launch_decode<D, 1, KVT>(p, w, stream); break;
    case 2: launch_decode<D, 2, KVT>(p, w, stream); break;
    case 3: launch_decode<D, 3, KVT>(p, w, stream); break;
    case 4: launch_decode<D, 4, KVT>(p, w, stream); break;
    case 5: launch_decode<D, 5, KVT>(p, w, stream); break;
    case 6: launch_decode<D, 6, KVT>(p, w, stream); break;
    case 7: launch_decode<D, 7, KVT>(p, w, stream); break;
    case 8: launch_decode<D, 8, KVT>(p, w, stream); break;
    default:
      throw Error(str_cat("unsupported GQA group size ", p.num_heads / p.kv.num_kv_heads));
  }
}

template <int D>
void dispatch_decode_kv(const AttentionParams& p, const DecodeAttentionWork& w,
                        cudaStream_t stream) {
  if (p.kv.dtype == KVDType::kFP16) {
    dispatch_decode_group<D, half>(p, w, stream);
  } else {
    dispatch_decode_group<D, int8_t>(p, w, stream);
  }
}

template <int D>
void launch_prefill(const AttentionParams& p, const PrefillAttentionWork& w, cudaStream_t stream) {
  dim3 grid(w.max_q_tiles, p.num_heads, w.num_seqs);
  if (p.kv.dtype == KVDType::kFP16) {
    prefill_attention_kernel<D, half><<<grid, kPrefillThreads, 0, stream>>>(p, w);
  } else {
    prefill_attention_kernel<D, int8_t><<<grid, kPrefillThreads, 0, stream>>>(p, w);
  }
  RL_CUDA_CHECK_LAUNCH();
}

}  // namespace

void paged_attention_decode(const AttentionParams& p, const DecodeAttentionWork& w,
                            cudaStream_t stream) {
  if (w.num_items <= 0) return;
  RL_CHECK(p.num_heads % p.kv.num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads");
  switch (p.kv.head_dim) {
    case 64: dispatch_decode_kv<64>(p, w, stream); break;
    case 128: dispatch_decode_kv<128>(p, w, stream); break;
    default: throw Error(str_cat("unsupported head_dim ", p.kv.head_dim));
  }
}

void paged_attention_prefill(const AttentionParams& p, const PrefillAttentionWork& w,
                             cudaStream_t stream) {
  if (w.num_seqs <= 0 || w.max_q_tiles <= 0) return;
  switch (p.kv.head_dim) {
    case 64: launch_prefill<64>(p, w, stream); break;
    case 128: launch_prefill<128>(p, w, stream); break;
    default: throw Error(str_cat("unsupported head_dim ", p.kv.head_dim));
  }
}

}  // namespace kernels
}  // namespace redline
