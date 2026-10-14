#include "cuda/model/llama.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/common.h"
#include "cuda/cuda_utils.h"
#include "cuda/kernels/kernels.h"
#include "cuda/nvtx.h"

namespace redline {

LlamaModel::LlamaModel(const ModelWeights* weights, LinearExecutor* linear,
                       const ModelRuntimeConfig& rc, MemoryLedger* ledger, const char* name)
    : weights_(weights), linear_(linear), rc_(rc), name_(name) {
  const ModelConfig& c = weights->config();
  attn_scale_ = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

  std::vector<float> cos_t, sin_t;
  rope_cos_sin_tables(c, rc.max_seq_len, &cos_t, &sin_t);
  rope_cos_ = ledger->alloc("activations", cos_t.size() * sizeof(float));
  rope_sin_ = ledger->alloc("activations", sin_t.size() * sizeof(float));
  RL_CUDA_CHECK(cudaMemcpy(rope_cos_.get(), cos_t.data(), cos_t.size() * 4, cudaMemcpyHostToDevice));
  RL_CUDA_CHECK(cudaMemcpy(rope_sin_.get(), sin_t.data(), sin_t.size() * 4, cudaMemcpyHostToDevice));

  const size_t T = rc.max_tokens, H = c.hidden_size, I = c.intermediate_size;
  const size_t h2 = sizeof(half);
  residual_ = ledger->alloc("activations", T * H * h2);
  normed_ = ledger->alloc("activations", T * H * h2);
  buf_a_ = ledger->alloc("activations", T * std::max<size_t>(c.qkv_size(), 2 * I) * h2);
  buf_b_ = ledger->alloc("activations", T * std::max<size_t>(c.q_size(), I) * h2);
  buf_c_ = ledger->alloc("activations", T * H * h2);
  logit_hidden_ = ledger->alloc("activations", static_cast<size_t>(rc.max_logit_rows) * H * h2);

  const size_t parts = static_cast<size_t>(max_partitions());
  const size_t ws_rows = static_cast<size_t>(rc.max_decode_items) * c.num_heads * parts;
  attn_ws_out_ = ledger->alloc("workspace", ws_rows * c.head_dim * sizeof(float));
  attn_ws_max_ = ledger->alloc("workspace", ws_rows * sizeof(float));
  attn_ws_sum_ = ledger->alloc("workspace", ws_rows * sizeof(float));
}

void LlamaModel::bind_kv_cache(const KVCache* kv) {
  const ModelConfig& c = weights_->config();
  RL_CHECK(kv->config().num_layers == c.num_layers && kv->config().head_dim == c.head_dim &&
               kv->config().num_kv_heads == c.num_kv_heads,
           "KV cache does not match model ", name_);
  kv_ = kv;
}

int LlamaModel::max_partitions() const {
  return static_cast<int>(ceil_div(rc_.max_seq_len, kernels::kDecodePartitionSize));
}

void LlamaModel::forward(const DeviceBatch& b, float* logits, bool static_partitions,
                         cudaStream_t s) {
  RL_NVTX_RANGE(name_, kNvtxStep);
  RL_CHECK(kv_ != nullptr, "forward: no KV cache bound to ", name_);
  const ModelConfig& c = weights_->config();
  const BatchShape& shape = b.shape;
  const int T = shape.num_tokens;
  RL_CHECK(T > 0 && T <= rc_.max_tokens, "forward: bad token count ", T);
  RL_CHECK(shape.num_logit_rows <= rc_.max_logit_rows, "forward: too many logits rows");
  RL_CHECK(shape.num_decode_items <= rc_.max_decode_items, "forward: too many decode items");
  RL_CHECK(shape.max_context <= rc_.max_seq_len, "forward: context exceeds max_seq_len");
  const int H = c.hidden_size;
  const float eps = static_cast<float>(c.rms_norm_eps);

  half* residual = residual_.as<half>();
  half* x = normed_.as<half>();
  half* a = buf_a_.as<half>();
  half* bb = buf_b_.as<half>();
  half* cc = buf_c_.as<half>();

  auto mark = [&](const char* op) {
    if (prof_ != nullptr) prof_->mark(op, s);
  };
  kernels::embedding_lookup(b.input_ids, weights_->tok_embeddings, T, H, residual, s);
  kernels::fused_add_rmsnorm(residual, nullptr, weights_->layers[0].attn_norm, T, H, eps, x, s);
  mark("embed+norm");

  kernels::AttentionParams ap;
  ap.q = a;
  ap.q_stride = c.qkv_size();
  ap.out = bb;
  ap.num_heads = c.num_heads;
  ap.scale = attn_scale_;
  ap.block_tables = b.block_tables;
  ap.max_blocks_per_seq = b.max_blocks_per_seq;
  ap.positions = b.positions;

  kernels::DecodeAttentionWork dw;
  dw.item_token = b.decode_token;
  dw.item_seq = b.decode_seq;
  dw.num_items = shape.num_decode_items;
  dw.max_partitions = static_partitions
                          ? max_partitions()
                          : static_cast<int>(ceil_div(std::max(shape.max_context, 1),
                                                      kernels::kDecodePartitionSize));
  dw.ws_out = attn_ws_out_.as<float>();
  dw.ws_max = attn_ws_max_.as<float>();
  dw.ws_sum = attn_ws_sum_.as<float>();

  kernels::PrefillAttentionWork pw;
  pw.seq_ids = b.prefill_seq;
  pw.q_start = b.seq_q_start;
  pw.q_len = b.seq_q_len;
  pw.ctx_len = b.seq_ctx_len;
  pw.num_seqs = shape.num_prefill_seqs;
  pw.max_q_tiles = shape.max_prefill_q_tiles;

  for (int l = 0; l < c.num_layers; ++l) {
    std::string layer_name;
    if (rc_.nvtx_per_layer) layer_name = "layer " + std::to_string(l);
    NvtxRange layer_range(rc_.nvtx_per_layer ? layer_name.c_str() : "layer", kNvtxLayer);
    const LayerWeights& lw = weights_->layers[l];

    linear_->forward(lw.wqkv, x, T, a, false, s);
    mark("qkv_proj");
    ap.kv = kv_->layer(l);
    kernels::rope_and_write_kv(a, b.positions, b.slot_mapping, T, c.num_heads,
                               rope_cos_.as<float>(), rope_sin_.as<float>(), ap.kv, s);
    mark("rope+kv_write");
    kernels::paged_attention_decode(ap, dw, s);
    mark("attention_decode");
    kernels::paged_attention_prefill(ap, pw, s);
    mark("attention_prefill");
    linear_->forward(lw.wo, bb, T, cc, false, s);
    mark("o_proj");
    kernels::fused_add_rmsnorm(residual, cc, lw.ffn_norm, T, H, eps, x, s);
    mark("add+rmsnorm");

    linear_->forward_gated(lw.w_gate_up, x, T, bb, a, s);  // bb = silu(gate) * up
    mark("gate_up+swiglu");
    linear_->forward(lw.w_down, bb, T, cc, false, s);
    mark("down_proj");
    const half* next_norm = l + 1 < c.num_layers ? weights_->layers[l + 1].attn_norm : weights_->norm;
    kernels::fused_add_rmsnorm(residual, cc, next_norm, T, H, eps, x, s);
    mark("add+rmsnorm");
  }

  const int R = shape.num_logit_rows;
  if (R > 0 && logits != nullptr) {
    RL_NVTX_RANGE("lm_head", kNvtxLayer);
    half* hrows = logit_hidden_.as<half>();
    kernels::gather_rows(x, b.logit_token, R, H, hrows, s);
    linear_->forward(weights_->lm_head, hrows, R, logits, true, s);
    mark("lm_head");
  }
}

}  // namespace redline
