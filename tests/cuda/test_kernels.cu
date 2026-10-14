// GPU kernel tests against straightforward CPU references: normalization,
// SwiGLU, embedding, RoPE + paged KV write, and paged attention (decode and
// prefill paths, FP16 and INT8 KV, several head dims / GQA groups, contexts
// that span multiple split-K partitions, and chunked prefill).
#include <algorithm>
#include <numeric>

#include "core/batch_builder.h"
#include "core/model_config.h"
#include "cuda/engine/device_batch.h"
#include "cuda/kernels/kernels.h"
#include "cuda/model/kv_cache.h"
#include "cuda_test_utils.h"

using namespace redline;
using namespace rltest;

TEST(kernel_fused_add_rmsnorm) {
  const int rows = 5, hidden = 512;
  auto res = round_half(randn(rows * hidden, 1.f, 1));
  auto del = round_half(randn(rows * hidden, 1.f, 2));
  auto w = round_half(randn(hidden, 0.2f, 3));
  for (auto& x : w) x += 1.f;
  DeviceVec<half> d_res(to_half(res)), d_del(to_half(del)), d_w(to_half(w)), d_out(rows * hidden);
  kernels::fused_add_rmsnorm(d_res.get(), d_del.get(), d_w.get(), rows, hidden, 1e-5f, d_out.get(), 0);
  std::vector<float> ref_res(res.size()), ref_out(res.size());
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int c = 0; c < hidden; ++c) {
      const size_t i = static_cast<size_t>(r) * hidden + c;
      ref_res[i] = __half2float(__float2half_rn(res[i] + del[i]));
      ss += ref_res[i] * ref_res[i];
    }
    const float inv = 1.f / std::sqrt(static_cast<float>(ss / hidden) + 1e-5f);
    for (int c = 0; c < hidden; ++c) {
      const size_t i = static_cast<size_t>(r) * hidden + c;
      ref_out[i] = __half2float(__float2half_rn(ref_res[i] * inv)) * w[c];
    }
  }
  EXPECT_CLOSE_VEC(to_float(d_res.download()), ref_res, 1e-3);
  EXPECT_CLOSE_VEC(to_float(d_out.download()), ref_out, 2e-3);

  // delta == nullptr: plain RMSNorm of the residual.
  kernels::fused_add_rmsnorm(d_res.get(), nullptr, d_w.get(), rows, hidden, 1e-5f, d_out.get(), 0);
  EXPECT_CLOSE_VEC(to_float(d_out.download()), ref_out, 2e-3);
}

TEST(kernel_silu_and_mul_and_embedding) {
  const int rows = 3, inter = 256, vocab = 50, hidden = 64;
  auto gu = round_half(randn(rows * 2 * inter, 2.f, 4));
  DeviceVec<half> d_gu(to_half(gu)), d_out(rows * inter);
  kernels::silu_and_mul(d_gu.get(), rows, inter, d_out.get(), 0);
  std::vector<float> ref(rows * inter);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < inter; ++i) {
      const float g = gu[r * 2 * inter + i], u = gu[r * 2 * inter + inter + i];
      ref[r * inter + i] = g / (1.f + std::exp(-g)) * u;
    }
  EXPECT_CLOSE_VEC(to_float(d_out.download()), ref, 3e-3);

  auto table = round_half(randn(vocab * hidden, 1.f, 5));
  std::vector<int32_t> ids = {3, 0, 49, 3};
  DeviceVec<half> d_table(to_half(table)), d_emb(ids.size() * hidden);
  DeviceVec<int32_t> d_ids(ids);
  kernels::embedding_lookup(d_ids.get(), d_table.get(), ids.size(), hidden, d_emb.get(), 0);
  auto emb = to_float(d_emb.download());
  for (size_t t = 0; t < ids.size(); ++t)
    for (int c = 0; c < hidden; ++c) EXPECT_NEAR(emb[t * hidden + c], table[ids[t] * hidden + c], 0);
}

namespace {

struct AttnCase {
  int hq, hkv, d;
  KVDType kv;
  std::vector<int> lens;
};

// CPU RoPE (rotate_half convention) on one head vector, rounding like the GPU.
void cpu_rope(float* v, int d, int pos, const std::vector<float>& cs, const std::vector<float>& sn) {
  const int half = d / 2;
  for (int i = 0; i < half; ++i) {
    const float c = cs[static_cast<size_t>(pos) * half + i], s = sn[static_cast<size_t>(pos) * half + i];
    const float x0 = v[i], x1 = v[i + half];
    v[i] = __half2float(__float2half_rn(x0 * c - x1 * s));
    v[i + half] = __half2float(__float2half_rn(x1 * c + x0 * s));
  }
}

void run_attention_case(const AttnCase& c, uint64_t seed) {
  const int bs = 16, G = c.hq / c.hkv, D = c.d, stride = (c.hq + 2 * c.hkv) * D;
  const int S = static_cast<int>(c.lens.size());
  const int T = std::accumulate(c.lens.begin(), c.lens.end(), 0);
  const int max_len = *std::max_element(c.lens.begin(), c.lens.end());
  ModelConfig mc;
  mc.head_dim = D;
  mc.rope_theta = 10000.0;
  std::vector<float> cs, sn;
  rope_cos_sin_tables(mc, max_len, &cs, &sn);
  DeviceVec<float> d_cos(cs), d_sin(sn);

  // Paged KV pool with shuffled block ids (block 0 reserved).
  int nblocks = 1;
  for (int L : c.lens) nblocks += (L + bs - 1) / bs;
  std::vector<int32_t> ids(nblocks - 1);
  std::iota(ids.begin(), ids.end(), 1);
  std::shuffle(ids.begin(), ids.end(), std::mt19937(seed));
  std::vector<std::vector<int32_t>> tables(S);
  size_t next = 0;
  const int max_blocks = (max_len + bs - 1) / bs;
  for (int s = 0; s < S; ++s)
    for (int b = 0; b < (c.lens[s] + bs - 1) / bs; ++b) tables[s].push_back(ids[next++]);
  KVCacheConfig kc;
  kc.num_layers = 1;
  kc.num_kv_heads = c.hkv;
  kc.head_dim = D;
  kc.block_size = bs;
  kc.num_blocks = nblocks;
  kc.dtype = c.kv;
  MemoryLedger ledger;
  KVCache cache(kc, &ledger);

  // Random qkv for every token (all sequences laid out back to back).
  auto qkv = round_half(randn(static_cast<size_t>(T) * stride, 1.f, seed + 1));
  DeviceVec<half> d_qkv(to_half(qkv));
  std::vector<int32_t> zeros(T, 0);

  auto build = [&](int decode_threshold, int chunk_start) {
    BatchBuilder bb(T, S, S, max_blocks, bs, decode_threshold);
    for (int s = 0; s < S; ++s) {
      const int start = std::min(chunk_start, c.lens[s] - 1);
      bb.add(zeros.data(), start, c.lens[s] - start, nullptr, tables[s], SamplingParams(), LogitsMode::kNone);
    }
    return bb;
  };
  auto upload = [&](const BatchBuilder& bb, DeviceVec<int32_t>& meta) {
    std::vector<int32_t> host(bb.max_meta_words());
    const BatchShape shape = bb.write(host.data());
    meta.upload(host);
    return DeviceBatch::bind(meta.get(), shape, max_blocks);
  };

  // Write K/V for all tokens; rope q in place.
  BatchBuilder full = build(0, 0);
  DeviceVec<int32_t> meta(full.max_meta_words());
  DeviceBatch b = upload(full, meta);
  kernels::rope_and_write_kv(d_qkv.get(), b.positions, b.slot_mapping, T, c.hq, d_cos.get(), d_sin.get(),
                             cache.layer(0), 0);

  // CPU reference.
  std::vector<float> ref(static_cast<size_t>(T) * c.hq * D, 0.f);
  std::vector<float> roped = qkv;
  {
    int t0 = 0;
    for (int s = 0; s < S; ++s) {
      for (int i = 0; i < c.lens[s]; ++i)
        for (int h = 0; h < c.hq + c.hkv; ++h) cpu_rope(&roped[static_cast<size_t>(t0 + i) * stride + h * D], D, i, cs, sn);
      for (int i = 0; i < c.lens[s]; ++i) {
        for (int h = 0; h < c.hq; ++h) {
          const float* q = &roped[static_cast<size_t>(t0 + i) * stride + h * D];
          std::vector<double> sc(i + 1);
          double mx = -1e30;
          for (int j = 0; j <= i; ++j) {
            const float* k = &roped[static_cast<size_t>(t0 + j) * stride + (c.hq + h / G) * D];
            double dot = 0;
            for (int x = 0; x < D; ++x) dot += q[x] * k[x];
            sc[j] = dot / std::sqrt(static_cast<double>(D));
            mx = std::max(mx, sc[j]);
          }
          double den = 0;
          for (auto& v : sc) den += (v = std::exp(v - mx));
          for (int x = 0; x < D; ++x) {
            double acc = 0;
            for (int j = 0; j <= i; ++j)
              acc += sc[j] * roped[static_cast<size_t>(t0 + j) * stride + (c.hq + c.hkv + h / G) * D + x];
            ref[(static_cast<size_t>(t0 + i) * c.hq + h) * D + x] = static_cast<float>(acc / den);
          }
        }
      }
      t0 += c.lens[s];
    }
  }
  // q heads were roped in place on the GPU.
  {
    auto gq = to_float(d_qkv.download());
    std::vector<float> a, r;
    for (int t = 0; t < T; ++t)
      for (int x = 0; x < c.hq * D; ++x) {
        a.push_back(gq[static_cast<size_t>(t) * stride + x]);
        r.push_back(roped[static_cast<size_t>(t) * stride + x]);
      }
    EXPECT_CLOSE_VEC(a, r, 2e-3);
  }

  const double tol = c.kv == KVDType::kFP16 ? 5e-3 : 3e-2;
  const int max_parts = (max_len + kernels::kDecodePartitionSize - 1) / kernels::kDecodePartitionSize;
  DeviceVec<float> ws_out(static_cast<size_t>(T) * c.hq * max_parts * D), ws_max(static_cast<size_t>(T) * c.hq * max_parts),
      ws_sum(static_cast<size_t>(T) * c.hq * max_parts);
  auto params = [&](const DeviceBatch& bt, half* out) {
    kernels::AttentionParams p;
    p.q = d_qkv.get();
    p.q_stride = stride;
    p.out = out;
    p.num_heads = c.hq;
    p.scale = 1.f / std::sqrt(static_cast<float>(D));
    p.kv = cache.layer(0);
    p.block_tables = bt.block_tables;
    p.max_blocks_per_seq = max_blocks;
    p.positions = bt.positions;
    return p;
  };

  // (1) Prefill path over all tokens.
  DeviceVec<half> out_p(static_cast<size_t>(T) * c.hq * D);
  {
    kernels::PrefillAttentionWork w;
    w.seq_ids = b.prefill_seq;
    w.q_start = b.seq_q_start;
    w.q_len = b.seq_q_len;
    w.ctx_len = b.seq_ctx_len;
    w.num_seqs = b.shape.num_prefill_seqs;
    w.max_q_tiles = b.shape.max_prefill_q_tiles;
    kernels::paged_attention_prefill(params(b, out_p.get()), w, 0);
    EXPECT_CLOSE_VEC(to_float(out_p.download()), ref, tol);
  }
  // (2) Decode path: every token is its own work item (split-K for long contexts).
  for (bool static_parts : {false, true}) {
    BatchBuilder dec = build(1 << 30, 0);
    DeviceVec<int32_t> meta2(dec.max_meta_words());
    DeviceBatch bd = upload(dec, meta2);
    DeviceVec<half> out_d(static_cast<size_t>(T) * c.hq * D);
    kernels::DecodeAttentionWork w;
    w.item_token = bd.decode_token;
    w.item_seq = bd.decode_seq;
    w.num_items = bd.shape.num_decode_items;
    w.max_partitions = static_parts ? max_parts : (bd.shape.max_context + kernels::kDecodePartitionSize - 1) /
                                                      kernels::kDecodePartitionSize;
    w.ws_out = ws_out.get();
    w.ws_max = ws_max.get();
    w.ws_sum = ws_sum.get();
    kernels::paged_attention_decode(params(bd, out_d.get()), w, 0);
    EXPECT_CLOSE_VEC(to_float(out_d.download()), ref, tol);
  }
  // (3) Chunked prefill: second half of each sequence attends to cached first half.
  {
    const int split = max_len / 2;
    BatchBuilder ch = build(0, split);
    DeviceVec<int32_t> meta3(ch.max_meta_words());
    DeviceBatch bc = upload(ch, meta3);
    // Gather the q rows of the chunk tokens into a compact buffer.
    std::vector<half> gq = d_qkv.download();
    std::vector<half> cq;
    std::vector<float> cref;
    int t0 = 0;
    for (int s = 0; s < S; ++s) {
      const int start = std::min(split, c.lens[s] - 1);
      for (int i = start; i < c.lens[s]; ++i) {
        cq.insert(cq.end(), gq.begin() + static_cast<size_t>(t0 + i) * stride,
                  gq.begin() + static_cast<size_t>(t0 + i + 1) * stride);
        cref.insert(cref.end(), ref.begin() + static_cast<size_t>(t0 + i) * c.hq * D,
                    ref.begin() + static_cast<size_t>(t0 + i + 1) * c.hq * D);
      }
      t0 += c.lens[s];
    }
    DeviceVec<half> d_cq(cq), out_c(cref.size());
    kernels::AttentionParams p = params(bc, out_c.get());
    p.q = d_cq.get();
    kernels::PrefillAttentionWork w;
    w.seq_ids = bc.prefill_seq;
    w.q_start = bc.seq_q_start;
    w.q_len = bc.seq_q_len;
    w.ctx_len = bc.seq_ctx_len;
    w.num_seqs = bc.shape.num_prefill_seqs;
    w.max_q_tiles = bc.shape.max_prefill_q_tiles;
    kernels::paged_attention_prefill(p, w, 0);
    EXPECT_CLOSE_VEC(to_float(out_c.download()), cref, tol);
  }
}

}  // namespace

TEST(kernel_attention_fp16_d128_gqa4) { run_attention_case({8, 2, 128, KVDType::kFP16, {37, 300, 5}}, 11); }
TEST(kernel_attention_fp16_d64_mha) { run_attention_case({4, 4, 64, KVDType::kFP16, {64, 129, 1}}, 12); }
TEST(kernel_attention_fp16_d128_gqa3_long) { run_attention_case({6, 2, 128, KVDType::kFP16, {700}}, 13); }
TEST(kernel_attention_fp16_d64_gqa8) { run_attention_case({8, 1, 64, KVDType::kFP16, {48, 17}}, 14); }
TEST(kernel_attention_int8_d128_gqa4) { run_attention_case({8, 2, 128, KVDType::kINT8, {33, 280}}, 15); }
TEST(kernel_attention_int8_d64_gqa2) { run_attention_case({4, 2, 64, KVDType::kINT8, {520}}, 16); }

TEST(kernel_rope_int8_kv_scales) {
  // INT8 cache: dequantized K must match the fp16 RoPE output within one step.
  const int hq = 2, hkv = 1, D = 64, bs = 16;
  ModelConfig mc;
  mc.head_dim = D;
  std::vector<float> cs, sn;
  rope_cos_sin_tables(mc, 8, &cs, &sn);
  DeviceVec<float> d_cos(cs), d_sin(sn);
  KVCacheConfig kc;
  kc.num_layers = 1;
  kc.num_kv_heads = hkv;
  kc.head_dim = D;
  kc.block_size = bs;
  kc.num_blocks = 2;
  kc.dtype = KVDType::kINT8;
  MemoryLedger ledger;
  KVCache cache(kc, &ledger);
  const int stride = (hq + 2 * hkv) * D;
  auto qkv = round_half(randn(stride, 1.f, 21));
  DeviceVec<half> d_qkv(to_half(qkv));
  DeviceVec<int32_t> pos(std::vector<int32_t>{5}), slot(std::vector<int32_t>{bs + 3});  // block 1, offset 3
  kernels::rope_and_write_kv(d_qkv.get(), pos.get(), slot.get(), 1, hq, d_cos.get(), d_sin.get(), cache.layer(0), 0);
  std::vector<float> k(qkv.begin() + hq * D, qkv.begin() + (hq + 1) * D);
  cpu_rope(k.data(), D, 5, cs, sn);
  std::vector<int8_t> kq(D);
  float scale = 0;
  RL_CUDA_CHECK(cudaMemcpy(kq.data(), static_cast<int8_t*>(cache.layer(0).k) + static_cast<size_t>(bs + 3) * D, D,
                           cudaMemcpyDeviceToHost));
  RL_CUDA_CHECK(cudaMemcpy(&scale, cache.layer(0).k_scale + bs + 3, 4, cudaMemcpyDeviceToHost));
  float amax = 0;
  for (float x : k) amax = std::max(amax, std::fabs(x));
  EXPECT_NEAR(scale, amax / 127.f, 1e-6);
  for (int i = 0; i < D; ++i) EXPECT_NEAR(kq[i] * scale, k[i], scale * 0.51);
}

