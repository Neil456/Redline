// Host-callable launchers for every CUDA kernel in the runtime. All launchers
// are asynchronous on the given stream and capturable into CUDA graphs (they
// never allocate or synchronize).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace redline {

enum class KVDType { kFP16 = 0, kINT8 = 1 };

// One layer of the paged KV cache.
//   k, v:            [num_blocks, num_kv_heads, block_size, head_dim]  (half or int8)
//   k_scale, v_scale [num_blocks, num_kv_heads, block_size] float     (INT8 only)
struct KVLayerView {
  void* k = nullptr;
  void* v = nullptr;
  float* k_scale = nullptr;
  float* v_scale = nullptr;
  int block_size = 16;
  int num_kv_heads = 0;
  int head_dim = 0;
  KVDType dtype = KVDType::kFP16;
};

// Per-request sampling parameters (struct-of-arrays, device pointers) and the
// mapping from logits rows to requests / generated positions.
struct SamplingParamsDevice {
  const float* temperature = nullptr;  // <= 0: greedy
  const int32_t* top_k = nullptr;      // <= 0: disabled
  const float* top_p = nullptr;        // >= 1: disabled
  const uint32_t* seed_lo = nullptr;
  const uint32_t* seed_hi = nullptr;
  const int32_t* param_row = nullptr;     // logits row -> param index (null: identity)
  const int32_t* token_of_row = nullptr;  // logits row -> token index in the batch
  const int32_t* positions = nullptr;     // token index -> position
  // The RNG counter of a row is the position of the token being generated
  // (positions[token_of_row[r]] + 1), so samples do not depend on batching.
};

// Independent random streams derived from the same (seed, position).
enum RngStream : uint32_t {
  kRngSample = 0,
  kRngAccept = 1,
  kRngResidual = 2,
  kRngDraft = 3,
};

// Statistics of one logits row under its sampling parameters.
struct RowStats {
  float max_logit;     // max raw logit
  float lse;           // log-sum-exp of raw logits (for log-probs)
  int32_t argmax;
  uint32_t threshold;  // order-preserving key: tokens with key >= threshold are kept
  float kept_mass;     // sum over kept tokens of exp((l - max_logit) / T)
  float inv_temp;      // 0 for greedy rows
};

namespace kernels {

// ---------------------------------------------------------------- elementwise
// input_ids[i] = src[i] >= 0 ? prev_sampled[src[i]] : input_ids[i]
void resolve_input_ids(int32_t* input_ids, const int32_t* src, const int32_t* prev_sampled, int n,
                       cudaStream_t stream);
void embedding_lookup(const int32_t* ids, const half* table, int n, int hidden, half* out,
                      cudaStream_t stream);
// residual += delta (skipped when delta == nullptr); out = rmsnorm(residual) * weight.
void fused_add_rmsnorm(half* residual, const half* delta, const half* weight, int rows,
                       int hidden, float eps, half* out, cudaStream_t stream);
// out[r, i] = silu(gate_up[r, i]) * gate_up[r, inter + i]
void silu_and_mul(const half* gate_up, int rows, int inter, half* out, cudaStream_t stream);
// Same, for gate/up columns interleaved as (gate_0, up_0, gate_1, up_1, ...).
void silu_and_mul_interleaved(const half* gate_up, int rows, int inter, half* out,
                              cudaStream_t stream);
// dst[i, :] = src[idx[i], :]
void gather_rows(const half* src, const int32_t* idx, int n, int cols, half* dst,
                 cudaStream_t stream);

// ------------------------------------------------------------------- rope / kv
// Applies RoPE in place to the q and k heads of the fused qkv activations
// [num_tokens, (Hq + 2 Hkv) * D] and writes k/v into the paged cache at
// slot_mapping[t] (block * block_size + offset; negative = skip).
void rope_and_write_kv(half* qkv, const int32_t* positions, const int32_t* slot_mapping,
                       int num_tokens, int num_heads, const float* cos_table,
                       const float* sin_table, const KVLayerView& kv, cudaStream_t stream);

// ------------------------------------------------------------------- attention
struct AttentionParams {
  const half* q = nullptr;  // query rows, stride q_stride elements per token
  int q_stride = 0;
  half* out = nullptr;      // [num_tokens, num_heads * head_dim]
  int num_heads = 0;
  float scale = 1.0f;
  KVLayerView kv;
  const int32_t* block_tables = nullptr;  // [num_seqs, max_blocks_per_seq]
  int max_blocks_per_seq = 0;
  const int32_t* positions = nullptr;     // [num_tokens]
};

// Decode / verification attention: one work item per query token, each
// attending to positions [0, positions[token]] of its sequence. Split-K over
// `partition_size` keys (flash-decoding); all query heads of a GQA group are
// processed together so each K/V vector is read once per group.
struct DecodeAttentionWork {
  const int32_t* item_token = nullptr;  // [num_items]
  const int32_t* item_seq = nullptr;    // [num_items]
  int num_items = 0;
  int max_partitions = 1;  // grid size; partitions beyond the context exit early
  float* ws_out = nullptr;  // [num_items, num_heads, max_partitions, head_dim]
  float* ws_max = nullptr;  // [num_items, num_heads, max_partitions]
  float* ws_sum = nullptr;
};
constexpr int kDecodePartitionSize = 256;
void paged_attention_decode(const AttentionParams& p, const DecodeAttentionWork& w,
                            cudaStream_t stream);

// Prefill attention: tiled (flash-attention style) causal attention for
// sequences processing several new tokens, attending to cached + new keys.
struct PrefillAttentionWork {
  const int32_t* seq_ids = nullptr;   // [num_seqs] indices into the arrays below
  const int32_t* q_start = nullptr;   // [*] first token row of each sequence
  const int32_t* q_len = nullptr;     // [*] number of new tokens
  const int32_t* ctx_len = nullptr;   // [*] total context after this step
  int num_seqs = 0;
  int max_q_tiles = 0;
};
constexpr int kPrefillTileQ = 32;
void paged_attention_prefill(const AttentionParams& p, const PrefillAttentionWork& w,
                             cudaStream_t stream);

// ---------------------------------------------------------------------- quant
// Dynamic per-row symmetric INT8 quantization of activations.
void quantize_rows_int8(const half* x, int rows, int cols, int8_t* q, float* row_scales,
                        cudaStream_t stream);
// out[m, n] = acc[m, n] * row_scale[m] * col_scale[n] (+ bias[n])
void dequantize_int32_gemm_output(const int32_t* acc, const float* row_scales,
                                  const half* col_scales, const half* bias, int rows, int cols,
                                  half* out, cudaStream_t stream);
// y[m, n] = y[m, n] * col_scale[n] * multiplier (+ bias[n]), in place.
void scale_columns(half* y, const half* col_scales, float multiplier, const half* bias, int rows,
                   int cols, cudaStream_t stream);
// y[m, n] += bias[n] for an fp32 output with row stride ldy (cuBLASLt has no fp16-bias epilogue
// for fp32 outputs).
void add_bias_f32(float* y, const half* bias, int rows, int cols, int ldy, cudaStream_t stream);
// Dequantize weight rows [row_begin, row_end) into out [row_end - row_begin, k].
void dequantize_weight_int8(const int8_t* w, const half* scales, int k, int row_begin,
                            int row_end, half* out, cudaStream_t stream);
void dequantize_weight_int4(const uint8_t* w, const half* scales, const half* offsets, int k,
                            int group_size, int row_begin, int row_end, half* out,
                            cudaStream_t stream);

// ------------------------------------------------------- weight-only GEMV
// y[m, n] = sum_k x[m, k] * dequant(w)[n, k] (+ bias[n]) for m <= kGemvW8MaxM (INT8) or
// m <= kGemvW4MaxM (INT4). Memory-bound decode kernels: weights streamed once with
// 16-byte loads and dequantized in registers. y is fp32 when y_fp32, else fp16.
constexpr int kGemvMaxM = 8;     // largest --gemv-max-m (rows served by one unchunked GEMV)
constexpr int kGemvW8MaxM = 16;  // rows per INT8 GEMV launch when chunking larger batches
constexpr int kGemvW4MaxM = 16;
void gemv_w8a16(const half* x, int m, const int8_t* w, const half* scales, const half* bias,
                int n, int k, void* y, bool y_fp32, cudaStream_t stream);
void gemv_w4a16(const half* x, int m, const uint8_t* w, const half* scales, const half* offsets,
                int group_size, const half* bias, int n, int k, void* y, bool y_fp32,
                cudaStream_t stream);
// Same kernel for a [gate; up] weight whose rows are interleaved (gate_i, up_i): writes the
// SwiGLU activation silu(gate) * up, fp16 [m, n / 2], from the GEMV epilogue.
void gemv_w4a16_swiglu(const half* x, int m, const uint8_t* w, const half* scales,
                       const half* offsets, int group_size, int n, int k, half* y,
                       cudaStream_t stream);

// ------------------------------------------------- W4A16 tensor-core GEMM
// y[m, n] = x[m, k] . dequant(w)^T (+ bias) on tensor cores for m <= kW4TcMaxM, with weights,
// scales and offsets in the tensor-core layout of core/int4_layout.h. With `silu_mul` the columns
// are interleaved (gate, up) pairs and y is the fp16 SwiGLU activation [m, n / 2].
// K is split across CTAs; partials go to `partials` and the last CTA per column block reduces
// them (deterministically) using `tickets`, which must start zeroed and are left zeroed.
constexpr int kW4TcMaxM = 64;
struct W4TcWorkspace {
  float* partials = nullptr;
  size_t partial_floats = 0;
  int* tickets = nullptr;
  int num_tickets = 0;
};
int w4a16_tc_splits(int n, int k, int group_size);
size_t w4a16_tc_workspace_floats(int max_m, int n, int k, int group_size);
// Bounds over every shape with n <= max_n (for a workspace shared by all layers).
size_t w4a16_tc_max_workspace_floats(int max_m, int max_n);
int w4a16_tc_max_tickets(int max_n);
void gemm_w4a16_tc(const half* x, int m, const uint8_t* w, const half* scales, const half* offsets,
                   int group_size, const half* bias, int n, int k, void* y, bool y_fp32,
                   bool silu_mul, const W4TcWorkspace& ws, cudaStream_t stream);
// Tensor-core layout -> fp16 weight rows [row_begin, row_end) (64-aligned), for prefill GEMMs.
void dequantize_weight_int4_tc(const uint8_t* w, const half* scales, const half* offsets, int k,
                               int group_size, int row_begin, int row_end, half* out,
                               cudaStream_t stream);

// ------------------------------------------------------------------- sampling
// Samples one token per logits row (greedy / temperature / top-k / top-p) with
// a counter-based RNG, so results are reproducible regardless of batching.
// Outputs the token, its raw log-probability and (optionally) row stats.
void sample_tokens(const float* logits, int rows, int vocab, const SamplingParamsDevice& params,
                   uint32_t rng_stream, int32_t* out_tokens, float* out_logprobs,
                   RowStats* out_stats, cudaStream_t stream);

// Row statistics only (used for speculative-decoding verification).
void compute_row_stats(const float* logits, int rows, int vocab, const SamplingParamsDevice& params,
                       RowStats* out_stats, cudaStream_t stream);

// Speculative decoding verification for `num_seqs` sequences, each with
// `num_draft` draft tokens. Target logits/stats row: seq * (num_draft + 1) + i.
// Draft logits/stats/token row: i * draft_stride + seq. `params` describes the
// target rows (param_row maps them to sequences). Accepts each draft token
// with probability min(1, p_target / p_draft) (exact match for greedy rows),
// resamples the first rejected position from max(0, p_target - p_draft) and
// appends a bonus token when every draft token is accepted.
// Writes out_tokens[seq, 0..num_draft] and out_count[seq] = accepted + 1.
void speculative_verify(const float* target_logits, const RowStats* target_stats,
                        const float* draft_logits, const RowStats* draft_stats,
                        const int32_t* draft_tokens, int num_seqs, int num_draft,
                        int draft_stride, int vocab, const SamplingParamsDevice& params,
                        int32_t* out_tokens, float* out_logprobs, int32_t* out_count,
                        cudaStream_t stream);

// out[r] = log_softmax(logits[r])[targets[r]]
void gather_logprobs(const float* logits, const int32_t* targets, int rows, int vocab, float* out,
                     cudaStream_t stream);

}  // namespace kernels
}  // namespace redline
