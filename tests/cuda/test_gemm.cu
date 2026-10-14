// GEMM tests: weight-only GEMV kernels, cuBLASLt FP16/INT8 wrappers and every
// LinearExecutor path (GEMV, chunked GEMV, dequantize+cuBLASLt, W8A8, CUTLASS).
#include <algorithm>
#include <type_traits>

#include "core/int4_layout.h"
#include "cpu/random_model.h"
#include "cuda/gemm/cublaslt_gemm.h"
#include "cuda/gemm/linear.h"
#include "cuda/kernels/kernels.h"
#include "cuda_test_utils.h"

using namespace redline;
using namespace rltest;

namespace {

struct QuantLinear {
  int n, k, g;
  QuantType q;
  std::vector<float> deq;  // dequantized reference weights [n, k]
  std::vector<float> bias;
  DeviceVec<uint8_t> w;
  DeviceVec<half> scales, offsets, d_bias;
  LinearWeight lw;
};

QuantLinear make_linear(int n, int k, QuantType q, int g, bool bias, uint64_t seed) {
  QuantLinear L{n, k, g, q, {}, {}, DeviceVec<uint8_t>(), DeviceVec<half>(), DeviceVec<half>(), DeviceVec<half>(), {}};
  auto w = randn(static_cast<size_t>(n) * k, 0.05f, seed);
  L.deq.resize(w.size());
  std::vector<uint8_t> bytes;
  std::vector<uint16_t> s16, o16;
  if (q == QuantType::kINT8) {
    std::vector<int8_t> qi;
    quantize_int8_rows(w, n, k, &qi, &s16);
    bytes.assign(reinterpret_cast<uint8_t*>(qi.data()), reinterpret_cast<uint8_t*>(qi.data()) + qi.size());
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < k; ++c) L.deq[static_cast<size_t>(r) * k + c] = qi[static_cast<size_t>(r) * k + c] * half_bits_to_float(s16[r]);
  } else {
    quantize_int4_groups(w, n, k, g, &bytes, &s16, &o16);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < k; ++c) {
        const uint8_t b = bytes[(static_cast<size_t>(r) * k + c) / 2];
        const int qv = (c & 1) ? (b >> 4) : (b & 0xF);
        const size_t gi = static_cast<size_t>(r) * (k / g) + c / g;
        L.deq[static_cast<size_t>(r) * k + c] = qv * half_bits_to_float(s16[gi]) + half_bits_to_float(o16[gi]);
      }
  }
  auto as_half = [](const std::vector<uint16_t>& v) {
    std::vector<half> h(v.size());
    std::memcpy(h.data(), v.data(), v.size() * 2);
    return h;
  };
  L.w = DeviceVec<uint8_t>(bytes);
  L.scales = DeviceVec<half>(as_half(s16));
  if (q == QuantType::kINT4) L.offsets = DeviceVec<half>(as_half(o16));
  if (bias) {
    L.bias = round_half(randn(n, 0.1f, seed + 7));
    L.d_bias = DeviceVec<half>(to_half(L.bias));
  }
  L.lw.name = "test";
  L.lw.qtype = q;
  L.lw.n = n;
  L.lw.k = k;
  L.lw.group_size = g;
  L.lw.weight = L.w.get();
  L.lw.scales = L.scales.get();
  L.lw.offsets = q == QuantType::kINT4 ? L.offsets.get() : nullptr;
  L.lw.bias = bias ? L.d_bias.get() : nullptr;
  return L;
}

std::vector<float> cpu_linear(const std::vector<float>& x, int m, const std::vector<float>& w, int n, int k,
                              const std::vector<float>& bias) {
  std::vector<float> y(static_cast<size_t>(m) * n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      double acc = bias.empty() ? 0.0 : bias[j];
      for (int c = 0; c < k; ++c) acc += static_cast<double>(x[static_cast<size_t>(i) * k + c]) * w[static_cast<size_t>(j) * k + c];
      y[static_cast<size_t>(i) * n + j] = static_cast<float>(acc);
    }
  return y;
}

// Copies an INT4 QuantLinear into the tensor-core layout (L.deq stays the logical reference).
struct TcCopy {
  DeviceVec<uint8_t> w;
  DeviceVec<half> s, o;
  LinearWeight lw;
};

TcCopy to_tc(const QuantLinear& L, bool interleave) {
  const auto bytes = L.w.download();
  auto bits = [](const std::vector<half>& h) {
    std::vector<uint16_t> v(h.size());
    std::memcpy(v.data(), h.data(), h.size() * 2);
    return v;
  };
  const auto s16 = bits(L.scales.download()), o16 = bits(L.offsets.download());
  std::vector<uint8_t> wt(bytes.size());
  std::vector<uint16_t> st(s16.size()), ot(o16.size());
  relayout_int4_weight_tc(bytes.data(), L.n, L.k, interleave, wt.data());
  relayout_int4_params_tc(s16.data(), L.n, L.k, L.g, interleave, st.data());
  relayout_int4_params_tc(o16.data(), L.n, L.k, L.g, interleave, ot.data());
  auto as_half = [](const std::vector<uint16_t>& v) {
    std::vector<half> h(v.size());
    std::memcpy(h.data(), v.data(), v.size() * 2);
    return h;
  };
  TcCopy t{DeviceVec<uint8_t>(wt), DeviceVec<half>(as_half(st)), DeviceVec<half>(as_half(ot)), L.lw};
  t.lw.weight = t.w.get();
  t.lw.scales = t.s.get();
  t.lw.offsets = t.o.get();
  t.lw.int4_tc = true;
  t.lw.interleaved = interleave;
  if (interleave) t.lw.bias = nullptr;
  return t;
}

// silu(gate_i) * up_i with the fp16 rounding points of the GPU paths.
std::vector<float> cpu_swiglu(const std::vector<float>& gu, int m, int inter) {
  auto r = [](float v) { return __half2float(__float2half_rn(v)); };
  std::vector<float> out(static_cast<size_t>(m) * inter);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < inter; ++j) {
      const float g = r(gu[static_cast<size_t>(i) * 2 * inter + j]);
      const float u = r(gu[static_cast<size_t>(i) * 2 * inter + inter + j]);
      out[static_cast<size_t>(i) * inter + j] = r(g / (1.f + std::exp(-g))) * u;
    }
  return out;
}

}  // namespace

TEST(gemv_weight_only_matches_cpu) {
  for (QuantType q : {QuantType::kINT8, QuantType::kINT4}) {
    for (int m : {1, 3, 8, 13, 16}) {
      for (bool f32 : {false, true}) {
        const int n = 97, k = 512, g = 64;  // odd n exercises the row tail
        QuantLinear L = make_linear(n, k, q, g, /*bias=*/true, 100 + m);
        auto x = round_half(randn(static_cast<size_t>(m) * k, 1.f, 200 + m));
        DeviceVec<half> dx(to_half(x));
        const auto ref = cpu_linear(x, m, L.deq, n, k, L.bias);
        if (f32) {
          DeviceVec<float> y(static_cast<size_t>(m) * n);
          if (q == QuantType::kINT8) {
            kernels::gemv_w8a16(dx.get(), m, reinterpret_cast<int8_t*>(L.w.get()), L.scales.get(), L.d_bias.get(), n, k, y.get(), true, 0);
          } else {
            kernels::gemv_w4a16(dx.get(), m, L.w.get(), L.scales.get(), L.offsets.get(), g, L.d_bias.get(), n, k, y.get(), true, 0);
          }
          EXPECT_CLOSE_VEC(y.download(), ref, 1e-4);
        } else {
          DeviceVec<half> y(static_cast<size_t>(m) * n);
          if (q == QuantType::kINT8) {
            kernels::gemv_w8a16(dx.get(), m, reinterpret_cast<int8_t*>(L.w.get()), L.scales.get(), L.d_bias.get(), n, k, y.get(), false, 0);
          } else {
            kernels::gemv_w4a16(dx.get(), m, L.w.get(), L.scales.get(), L.offsets.get(), g, L.d_bias.get(), n, k, y.get(), false, 0);
          }
          EXPECT_CLOSE_VEC(to_float(y.download()), ref, 2e-3);
        }
      }
    }
  }
}

TEST(w4a16_tensor_core_gemm_matches_cpu) {
  struct Shape {
    int n, k, g;
  };
  // splits > 1 (deterministic last-CTA reduction) and a single-split shape.
  for (const Shape sh : {Shape{384, 512, 64}, Shape{256, 1024, 128}, Shape{256, 64, 64}}) {
    QuantLinear L = make_linear(sh.n, sh.k, QuantType::kINT4, sh.g, /*bias=*/true, 500 + sh.k);
    TcCopy T = to_tc(L, false);
    DeviceVec<float> partials(std::max<size_t>(1, kernels::w4a16_tc_workspace_floats(64, sh.n, sh.k, sh.g)));
    DeviceVec<int> tickets(64);
    RL_CUDA_CHECK(cudaMemset(tickets.get(), 0, 64 * sizeof(int)));
    kernels::W4TcWorkspace ws;
    ws.partials = partials.get();
    ws.partial_floats = partials.size();
    ws.tickets = tickets.get();
    ws.num_tickets = 64;
    for (int m : {1, 7, 16, 17, 40, 64}) {
      auto x = round_half(randn(static_cast<size_t>(m) * sh.k, 1.f, 600 + m));
      DeviceVec<half> dx(to_half(x));
      const auto ref = cpu_linear(x, m, L.deq, sh.n, sh.k, L.bias);
      DeviceVec<half> y16(static_cast<size_t>(m) * sh.n);
      kernels::gemm_w4a16_tc(dx.get(), m, T.w.get(), T.s.get(), T.o.get(), sh.g, L.d_bias.get(), sh.n, sh.k,
                             y16.get(), false, false, ws, 0);
      EXPECT_CLOSE_VEC(to_float(y16.download()), ref, 4e-3);
      DeviceVec<float> y32(static_cast<size_t>(m) * sh.n);
      kernels::gemm_w4a16_tc(dx.get(), m, T.w.get(), T.s.get(), T.o.get(), sh.g, L.d_bias.get(), sh.n, sh.k,
                             y32.get(), true, false, ws, 0);
      const auto got = y32.download();
      EXPECT_CLOSE_VEC(got, ref, 2e-3);
      // Same inputs again: the split-K reduction is deterministic and the tickets were reset.
      kernels::gemm_w4a16_tc(dx.get(), m, T.w.get(), T.s.get(), T.o.get(), sh.g, L.d_bias.get(), sh.n, sh.k,
                             y32.get(), true, false, ws, 0);
      const auto again = y32.download();
      for (size_t i = 0; i < got.size(); ++i) EXPECT_EQ(got[i], again[i]);
    }
    EXPECT_TRUE(tickets.download() == std::vector<int>(64, 0));
  }
}

TEST(w4a16_tensor_core_fused_swiglu_and_dequant) {
  const int inter = 256, n = 2 * inter, k = 256, g = 64;
  QuantLinear L = make_linear(n, k, QuantType::kINT4, g, /*bias=*/false, 700);
  TcCopy T = to_tc(L, /*interleave=*/true);
  DeviceVec<float> partials(std::max<size_t>(1, kernels::w4a16_tc_workspace_floats(64, n, k, g)));
  DeviceVec<int> tickets(64);
  RL_CUDA_CHECK(cudaMemset(tickets.get(), 0, 64 * sizeof(int)));
  kernels::W4TcWorkspace ws{partials.get(), partials.size(), tickets.get(), 64};
  for (int m : {1, 9, 33}) {
    auto x = round_half(randn(static_cast<size_t>(m) * k, 1.f, 800 + m));
    DeviceVec<half> dx(to_half(x)), out(static_cast<size_t>(m) * inter);
    kernels::gemm_w4a16_tc(dx.get(), m, T.w.get(), T.s.get(), T.o.get(), g, nullptr, n, k, out.get(), false,
                           /*silu_mul=*/true, ws, 0);
    const auto ref = cpu_swiglu(cpu_linear(x, m, L.deq, n, k, {}), m, inter);
    EXPECT_CLOSE_VEC(to_float(out.download()), ref, 6e-3);
  }
  // Prefill path: the tc-layout dequantizer reproduces the (interleaved) weight rows exactly.
  DeviceVec<half> rows(static_cast<size_t>(128) * k);
  kernels::dequantize_weight_int4_tc(T.w.get(), T.s.get(), T.o.get(), k, g, 64, 192, rows.get(), 0);
  const auto got = to_float(rows.download());
  for (int r = 0; r < 128; ++r) {
    const int src = int4_tc_source_row(64 + r, n, true);
    for (int c = 0; c < k; ++c) {
      EXPECT_EQ(got[static_cast<size_t>(r) * k + c],
                __half2float(__float2half_rn(L.deq[static_cast<size_t>(src) * k + c])));
    }
  }
}

TEST(linear_executor_int4_tensor_core_paths) {
  const int inter = 192, k = 256, g = 64;
  LinearOptions lo;
  lo.int4_tensor_cores = true;
  lo.max_m = 256;
  lo.max_k = 2 * inter;
  lo.max_n = 2 * inter;
  lo.dequant_scratch = 64 * k * 2;  // several dequant tiles
  MemoryLedger ledger;
  LinearExecutor exec(lo, &ledger);
  QuantLinear L = make_linear(2 * inter, k, QuantType::kINT4, g, /*bias=*/true, 900);
  TcCopy plain = to_tc(L, false), gated = to_tc(L, true);
  for (int m : {5, 40, 100}) {  // tensor-core GEMM, tensor-core GEMM, dequant + cuBLASLt
    auto x = round_half(randn(static_cast<size_t>(m) * k, 1.f, 1000 + m));
    DeviceVec<half> dx(to_half(x)), y(static_cast<size_t>(m) * 2 * inter), act(static_cast<size_t>(m) * inter);
    exec.forward(plain.lw, dx.get(), m, y.get(), false, 0);
    EXPECT_CLOSE_VEC(to_float(y.download()), cpu_linear(x, m, L.deq, 2 * inter, k, L.bias), 4e-3);
    exec.forward_gated(gated.lw, dx.get(), m, act.get(), y.get(), 0);
    const auto ref = cpu_swiglu(cpu_linear(x, m, L.deq, 2 * inter, k, {}), m, inter);
    EXPECT_CLOSE_VEC(to_float(act.download()), ref, 6e-3);
  }
}

TEST(gemv_w4a16_swiglu_interleaved_rows) {
  const int inter = 192, n = 2 * inter, k = 256, g = 64;
  QuantLinear L = make_linear(n, k, QuantType::kINT4, g, /*bias=*/false, 1100);
  // Rows (gate_i, up_i) interleaved, as the loader stores INT4 [gate; up] in the GEMV layout.
  auto interleave = [&](const auto& v, size_t row_elems) {
    std::remove_const_t<std::remove_reference_t<decltype(v)>> out(v.size());
    for (int i = 0; i < inter; ++i) {
      std::copy_n(v.begin() + i * row_elems, row_elems, out.begin() + (2 * i) * row_elems);
      std::copy_n(v.begin() + (inter + i) * row_elems, row_elems, out.begin() + (2 * i + 1) * row_elems);
    }
    return out;
  };
  DeviceVec<uint8_t> w(interleave(L.w.download(), k / 2));
  DeviceVec<half> sc(interleave(L.scales.download(), k / g)), of(interleave(L.offsets.download(), k / g));
  LinearWeight lw = L.lw;
  lw.weight = w.get();
  lw.scales = sc.get();
  lw.offsets = of.get();
  lw.bias = nullptr;
  lw.interleaved = true;
  LinearOptions lo;
  lo.int4_tensor_cores = false;
  lo.max_m = 256;
  lo.max_k = k;
  lo.max_n = n;
  lo.dequant_scratch = 64 * k * 2;
  MemoryLedger ledger;
  LinearExecutor exec(lo, &ledger);
  for (int m : {1, 3, 8, 20, 100}) {  // fused GEMV epilogue, chunked GEMV + kernel, dequant + kernel
    auto x = round_half(randn(static_cast<size_t>(m) * k, 1.f, 1200 + m));
    DeviceVec<half> dx(to_half(x)), act(static_cast<size_t>(m) * inter), scratch(static_cast<size_t>(m) * n);
    exec.forward_gated(lw, dx.get(), m, act.get(), scratch.get(), 0);
    const auto ref = cpu_swiglu(cpu_linear(x, m, L.deq, n, k, {}), m, inter);
    EXPECT_CLOSE_VEC(to_float(act.download()), ref, 6e-3);
  }
}

TEST(cublaslt_fp16_and_int8) {
  MemoryLedger ledger;
  CublasLtGemm lt(8u << 20, &ledger);
  const int m = 37, n = 200, k = 128;
  auto x = round_half(randn(static_cast<size_t>(m) * k, 1.f, 1));
  auto w = round_half(randn(static_cast<size_t>(n) * k, 0.1f, 2));
  auto b = round_half(randn(n, 0.1f, 3));
  DeviceVec<half> dx(to_half(x)), dw(to_half(w)), db(to_half(b));
  const auto ref = cpu_linear(x, m, w, n, k, b);
  DeviceVec<float> y32(static_cast<size_t>(m) * n);
  lt.gemm_fp16(dx.get(), dw.get(), db.get(), y32.get(), true, m, n, k, n, 0);
  EXPECT_CLOSE_VEC(y32.download(), ref, 1e-3);
  DeviceVec<half> y16(static_cast<size_t>(m) * n);
  lt.gemm_fp16(dx.get(), dw.get(), nullptr, y16.get(), false, m, n, k, n, 0);
  EXPECT_CLOSE_VEC(to_float(y16.download()), cpu_linear(x, m, w, n, k, {}), 3e-3);

  // INT8 x INT8 -> INT32 is exact.
  std::vector<int8_t> xi(static_cast<size_t>(m) * k), wi(static_cast<size_t>(n) * k);
  std::mt19937 rng(4);
  for (auto& v : xi) v = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
  for (auto& v : wi) v = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
  DeviceVec<int8_t> dxi(xi), dwi(wi);
  DeviceVec<int32_t> yi(static_cast<size_t>(m) * n);
  lt.gemm_int8(dxi.get(), dwi.get(), yi.get(), m, n, k, 0);
  auto got = yi.download();
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      int32_t acc = 0;
      for (int c = 0; c < k; ++c) acc += xi[static_cast<size_t>(i) * k + c] * wi[static_cast<size_t>(j) * k + c];
      EXPECT_EQ(got[static_cast<size_t>(i) * n + j], acc);
    }
}

TEST(linear_executor_all_paths) {
  struct Path {
    QuantType q;
    PrefillGemm backend;
    double tol;
  };
  std::vector<Path> paths = {
      {QuantType::kINT8, PrefillGemm::kDequant, 3e-3},
      {QuantType::kINT8, PrefillGemm::kW8A8, 3e-2},  // activations quantized too
      {QuantType::kINT4, PrefillGemm::kDequant, 3e-3},
  };
  if (LinearExecutor::cutlass_available()) paths.push_back({QuantType::kINT8, PrefillGemm::kCutlass, 3e-3});
  const int n = 384, k = 256, g = 64;
  for (const Path& p : paths) {
    LinearOptions lo;
    lo.int8_prefill = p.q == QuantType::kINT8 ? p.backend : PrefillGemm::kAuto;
    lo.int4_prefill = p.q == QuantType::kINT4 ? p.backend : PrefillGemm::kAuto;
    lo.max_m = 256;
    lo.max_k = k;
    lo.max_n = n;
    lo.dequant_scratch = 64 * k * 2;  // forces several dequant tiles
    MemoryLedger ledger;
    LinearExecutor exec(lo, &ledger);
    QuantLinear L = make_linear(n, k, p.q, g, true, 300);
    for (int m : {5, 20, 40, 100}) {  // GEMV, chunked GEMV / tensor-core path, batched GEMM
      auto x = round_half(randn(static_cast<size_t>(m) * k, 1.f, 400 + m));
      DeviceVec<half> dx(to_half(x)), y(static_cast<size_t>(m) * n);
      exec.forward(L.lw, dx.get(), m, y.get(), false, 0);
      const double tol = m <= lo.gemv_max_m ? 3e-3 : p.tol;
      EXPECT_CLOSE_VEC(to_float(y.download()), cpu_linear(x, m, L.deq, n, k, L.bias), tol);
      DeviceVec<float> y32(static_cast<size_t>(m) * n);
      exec.forward(L.lw, dx.get(), m, y32.get(), true, 0);  // fp32 output (lm_head)
      EXPECT_CLOSE_VEC(y32.download(), cpu_linear(x, m, L.deq, n, k, L.bias), 3e-3);
    }
  }
}
