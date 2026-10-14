// GEMM tests: weight-only GEMV kernels, cuBLASLt FP16/INT8 wrappers and every
// LinearExecutor path (GEMV, chunked GEMV, dequantize+cuBLASLt, W8A8, CUTLASS).
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

}  // namespace

TEST(gemv_weight_only_matches_cpu) {
  for (QuantType q : {QuantType::kINT8, QuantType::kINT4}) {
    for (int m : {1, 3, 8, 13, 16}) {
      if (q == QuantType::kINT8 && m > kernels::kGemvMaxM) continue;
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
