// GPU token sampling and speculative-decoding verification.
//
// One CTA per logits row. Top-k and top-p thresholds are found with exact
// 4-pass radix selection on order-preserving float keys (counts for top-k,
// probability mass for top-p), so no vocabulary sort is needed. The token is
// then drawn with the Gumbel-max trick over the kept set using a Philox
// counter-based RNG keyed by (seed, position), which makes sampling
// independent of batch composition.
#include <cfloat>

#include "cuda/cuda_utils.h"
#include "cuda/kernels/common.cuh"
#include "cuda/kernels/kernels.h"

namespace redline {
namespace kernels {
namespace {

constexpr int kSampleThreads = 1024;

struct RowParams {
  float temperature;
  int top_k;
  float top_p;
  uint32_t seed_lo, seed_hi;
  uint32_t position;  // position of the token being generated
};

struct SampleShared {
  float scratch[32];
  ArgMax am[32];
  uint32_t hist[256];
  float mass[256];
  uint32_t prefix;
  uint32_t count_rem;
  float mass_rem;
};

__device__ RowParams load_params(const SamplingParamsDevice& p, int row) {
  const int pr = p.param_row ? p.param_row[row] : row;
  RowParams rp;
  rp.temperature = p.temperature[pr];
  rp.top_k = p.top_k[pr];
  rp.top_p = p.top_p[pr];
  rp.seed_lo = p.seed_lo[pr];
  rp.seed_hi = p.seed_hi[pr];
  const int tok = p.token_of_row ? p.token_of_row[row] : row;
  rp.position = p.positions ? static_cast<uint32_t>(p.positions[tok] + 1) : 0u;
  return rp;
}

// Key of the k-th largest logit (exact, ties included).
__device__ uint32_t radix_select_topk(const float* l, int vocab, int k, SampleShared& sh) {
  uint32_t prefix = 0, mask = 0, krem = static_cast<uint32_t>(k);
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += blockDim.x) sh.hist[b] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
      const uint32_t key = float_to_key(l[i]);
      if ((key & mask) == prefix) atomicAdd(&sh.hist[(key >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t cum = 0, sel = 0;
      for (int b = 255; b >= 0; --b) {
        const uint32_t c = sh.hist[b];
        if (cum + c >= krem) {
          sel = static_cast<uint32_t>(b);
          break;
        }
        cum += c;
      }
      sh.count_rem = krem - cum;
      sh.prefix = prefix | (sel << shift);
    }
    __syncthreads();
    prefix = sh.prefix;
    krem = sh.count_rem;
    mask |= 0xFFu << shift;
  }
  return prefix;
}

// Largest key threshold whose kept probability mass reaches `target`, among
// tokens with key >= min_key.
__device__ uint32_t radix_select_mass(const float* l, int vocab, float max_logit, float inv_t,
                                      uint32_t min_key, float target, SampleShared& sh) {
  uint32_t prefix = 0, mask = 0;
  float rem = target;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += blockDim.x) sh.mass[b] = 0.f;
    __syncthreads();
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
      const uint32_t key = float_to_key(l[i]);
      if (key >= min_key && (key & mask) == prefix) {
        atomicAdd(&sh.mass[(key >> shift) & 0xFFu], __expf((l[i] - max_logit) * inv_t));
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float cum = 0.f;
      int sel = -1, lowest_nonzero = 0;
      for (int b = 255; b >= 0; --b) {
        const float c = sh.mass[b];
        if (c > 0.f) lowest_nonzero = b;
        if (c > 0.f && cum + c >= rem) {
          sel = b;
          break;
        }
        cum += c;
      }
      if (sel < 0) {
        // Rounding left the target unreachable: keep everything that remains.
        sel = lowest_nonzero;
        sh.mass_rem = FLT_MAX;
      } else {
        sh.mass_rem = rem - cum;
      }
      sh.prefix = prefix | (static_cast<uint32_t>(sel) << shift);
    }
    __syncthreads();
    prefix = sh.prefix;
    rem = sh.mass_rem;
    mask |= 0xFFu << shift;
  }
  return prefix;
}

__device__ RowStats row_stats(const float* l, int vocab, const RowParams& rp, SampleShared& sh) {
  ArgMax a{-FLT_MAX, 0};
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) a = argmax_combine(a, ArgMax{l[i], i});
  a = block_reduce_argmax(a, sh.am);
  const float mx = a.v;
  float s = 0.f;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) s += __expf(l[i] - mx);
  s = block_reduce_sum(s, sh.scratch);

  RowStats st;
  st.max_logit = mx;
  st.lse = mx + __logf(s);
  st.argmax = a.i;
  st.threshold = 0;
  st.kept_mass = 1.f;
  st.inv_temp = 0.f;
  if (rp.temperature <= 0.f) return st;

  const float inv_t = 1.f / rp.temperature;
  uint32_t thr = 0;
  if (rp.top_k > 0 && rp.top_k < vocab) thr = radix_select_topk(l, vocab, rp.top_k, sh);
  if (rp.top_p < 1.f) {
    float total = 0.f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
      if (float_to_key(l[i]) >= thr) total += __expf((l[i] - mx) * inv_t);
    }
    total = block_reduce_sum(total, sh.scratch);
    const uint32_t thr_p = radix_select_mass(l, vocab, mx, inv_t, thr, rp.top_p * total, sh);
    thr = max(thr, thr_p);
  }
  float kept = 0.f;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    if (float_to_key(l[i]) >= thr) kept += __expf((l[i] - mx) * inv_t);
  }
  kept = block_reduce_sum(kept, sh.scratch);
  st.threshold = thr;
  st.kept_mass = kept;
  st.inv_temp = inv_t;
  return st;
}

// Probability of a token with raw logit `logit` under the row's warped distribution.
__device__ __forceinline__ float warped_prob(const RowStats& st, float logit) {
  if (float_to_key(logit) < st.threshold) return 0.f;
  return __expf((logit - st.max_logit) * st.inv_temp) / st.kept_mass;
}

// Gumbel-max draw from the kept set of a (non-greedy) row.
__device__ int gumbel_sample(const float* l, int vocab, const RowStats& st, const RowParams& rp,
                             uint32_t stream, SampleShared& sh) {
  ArgMax best{-FLT_MAX, 0x7fffffff};
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    if (float_to_key(l[i]) < st.threshold) continue;
    const float u = uniform01(rp.seed_lo, rp.seed_hi, rp.position, stream, static_cast<uint32_t>(i));
    best = argmax_combine(best, ArgMax{(l[i] - st.max_logit) * st.inv_temp + gumbel(u), i});
  }
  best = block_reduce_argmax(best, sh.am);
  return best.i == 0x7fffffff ? st.argmax : best.i;
}

__global__ void __launch_bounds__(kSampleThreads)
sample_kernel(const float* logits, int vocab, SamplingParamsDevice params, uint32_t rng_stream,
              int32_t* out_tokens, float* out_logprobs, RowStats* out_stats) {
  __shared__ SampleShared sh;
  const int row = blockIdx.x;
  const float* l = logits + static_cast<size_t>(row) * vocab;
  const RowParams rp = load_params(params, row);
  const RowStats st = row_stats(l, vocab, rp, sh);
  const int token = rp.temperature <= 0.f ? st.argmax : gumbel_sample(l, vocab, st, rp, rng_stream, sh);
  if (threadIdx.x == 0) {
    out_tokens[row] = token;
    if (out_logprobs) out_logprobs[row] = l[token] - st.lse;
    if (out_stats) out_stats[row] = st;
  }
}

__global__ void __launch_bounds__(kSampleThreads)
row_stats_kernel(const float* logits, int vocab, SamplingParamsDevice params, RowStats* out_stats) {
  __shared__ SampleShared sh;
  const int row = blockIdx.x;
  const float* l = logits + static_cast<size_t>(row) * vocab;
  const RowStats st = row_stats(l, vocab, load_params(params, row), sh);
  if (threadIdx.x == 0) out_stats[row] = st;
}

__global__ void __launch_bounds__(kSampleThreads)
speculative_verify_kernel(const float* target_logits, const RowStats* target_stats,
                          const float* draft_logits, const RowStats* draft_stats,
                          const int32_t* draft_tokens, int num_draft, int draft_stride,
                          int vocab, SamplingParamsDevice params, int32_t* out_tokens,
                          float* out_logprobs, int32_t* out_count) {
  __shared__ SampleShared sh;
  const int seq = blockIdx.x;
  const int out_base = seq * (num_draft + 1);

  // Every thread evaluates the same acceptance decisions, so control flow
  // stays uniform across the block.
  for (int i = 0; i < num_draft; ++i) {
    const int trow = seq * (num_draft + 1) + i;
    const int drow = i * draft_stride + seq;
    const float* tl = target_logits + static_cast<size_t>(trow) * vocab;
    const RowStats ts = target_stats[trow];
    const RowParams rp = load_params(params, trow);
    const int x = draft_tokens[drow];
    bool accept;
    if (rp.temperature <= 0.f) {
      accept = x == ts.argmax;
    } else {
      const float* dl = draft_logits + static_cast<size_t>(drow) * vocab;
      const RowStats ds = draft_stats[drow];
      const float pt = warped_prob(ts, tl[x]);
      const float pd = warped_prob(ds, dl[x]);
      const float u = uniform01(rp.seed_lo, rp.seed_hi, rp.position, kRngAccept, 0);
      accept = pd > 0.f ? (u * pd < pt) : (pt > 0.f);
    }
    if (accept) {
      if (threadIdx.x == 0) {
        out_tokens[out_base + i] = x;
        out_logprobs[out_base + i] = tl[x] - ts.lse;
      }
      continue;
    }
    // Rejected: resample from the residual distribution max(0, p_t - p_d).
    int token;
    if (rp.temperature <= 0.f) {
      token = ts.argmax;
    } else {
      const float* dl = draft_logits + static_cast<size_t>(drow) * vocab;
      const RowStats ds = draft_stats[drow];
      ArgMax best{-FLT_MAX, 0x7fffffff};
      for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        const float r = warped_prob(ts, tl[v]) - warped_prob(ds, dl[v]);
        if (r <= 0.f) continue;
        const float u = uniform01(rp.seed_lo, rp.seed_hi, rp.position, kRngResidual, static_cast<uint32_t>(v));
        best = argmax_combine(best, ArgMax{__logf(r) + gumbel(u), v});
      }
      best = block_reduce_argmax(best, sh.am);
      token = best.i != 0x7fffffff ? best.i : gumbel_sample(tl, vocab, ts, rp, kRngResidual, sh);
    }
    if (threadIdx.x == 0) {
      out_tokens[out_base + i] = token;
      out_logprobs[out_base + i] = tl[token] - ts.lse;
      out_count[seq] = i + 1;
    }
    return;
  }
  // All drafts accepted: bonus token from the last target row.
  const int trow = seq * (num_draft + 1) + num_draft;
  const float* tl = target_logits + static_cast<size_t>(trow) * vocab;
  const RowStats ts = target_stats[trow];
  const RowParams rp = load_params(params, trow);
  const int token = rp.temperature <= 0.f ? ts.argmax : gumbel_sample(tl, vocab, ts, rp, kRngSample, sh);
  if (threadIdx.x == 0) {
    out_tokens[out_base + num_draft] = token;
    out_logprobs[out_base + num_draft] = tl[token] - ts.lse;
    out_count[seq] = num_draft + 1;
  }
}

__global__ void gather_logprobs_kernel(const float* logits, const int32_t* targets, int vocab,
                                       float* out) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  const float* l = logits + static_cast<size_t>(row) * vocab;
  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) m = fmaxf(m, l[i]);
  m = block_reduce_max(m, scratch);
  float s = 0.f;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) s += __expf(l[i] - m);
  s = block_reduce_sum(s, scratch);
  if (threadIdx.x == 0) out[row] = l[targets[row]] - (m + __logf(s));
}

}  // namespace

void sample_tokens(const float* logits, int rows, int vocab, const SamplingParamsDevice& params,
                   uint32_t rng_stream, int32_t* out_tokens, float* out_logprobs,
                   RowStats* out_stats, cudaStream_t stream) {
  if (rows <= 0) return;
  sample_kernel<<<rows, kSampleThreads, 0, stream>>>(logits, vocab, params, rng_stream, out_tokens,
                                                     out_logprobs, out_stats);
  RL_CUDA_CHECK_LAUNCH();
}

void compute_row_stats(const float* logits, int rows, int vocab, const SamplingParamsDevice& params,
                       RowStats* out_stats, cudaStream_t stream) {
  if (rows <= 0) return;
  row_stats_kernel<<<rows, kSampleThreads, 0, stream>>>(logits, vocab, params, out_stats);
  RL_CUDA_CHECK_LAUNCH();
}

void speculative_verify(const float* target_logits, const RowStats* target_stats,
                        const float* draft_logits, const RowStats* draft_stats,
                        const int32_t* draft_tokens, int num_seqs, int num_draft,
                        int draft_stride, int vocab, const SamplingParamsDevice& params,
                        int32_t* out_tokens, float* out_logprobs, int32_t* out_count,
                        cudaStream_t stream) {
  if (num_seqs <= 0) return;
  RL_CHECK(draft_stride >= num_seqs, "draft_stride must be >= num_seqs");
  speculative_verify_kernel<<<num_seqs, kSampleThreads, 0, stream>>>(
      target_logits, target_stats, draft_logits, draft_stats, draft_tokens, num_draft,
      draft_stride, vocab, params, out_tokens, out_logprobs, out_count);
  RL_CUDA_CHECK_LAUNCH();
}

void gather_logprobs(const float* logits, const int32_t* targets, int rows, int vocab, float* out,
                     cudaStream_t stream) {
  if (rows <= 0) return;
  gather_logprobs_kernel<<<rows, 512, 0, stream>>>(logits, targets, vocab, out);
  RL_CUDA_CHECK_LAUNCH();
}

}  // namespace kernels
}  // namespace redline
