// Sampling tests: greedy, top-k / top-p kept sets, empirical distributions
// vs the exact warped distribution, determinism, log-probs, and speculative
// verification (greedy exactness and lossless rejection sampling).
#include <algorithm>
#include <map>

#include "cuda/kernels/kernels.h"
#include "cuda_test_utils.h"

using namespace redline;
using namespace rltest;

namespace {

struct HostParams {
  std::vector<float> temperature, top_p;
  std::vector<int32_t> top_k;
  std::vector<uint32_t> seed_lo, seed_hi;
  std::vector<int32_t> positions;  // per row (token_of_row = identity)
};

struct DeviceParams {
  DeviceVec<float> temperature, top_p;
  DeviceVec<int32_t> top_k, positions, param_row;
  DeviceVec<uint32_t> seed_lo, seed_hi;
  SamplingParamsDevice view;
  explicit DeviceParams(const HostParams& h, const std::vector<int32_t>& param_row = {})
      : temperature(h.temperature), top_p(h.top_p), top_k(h.top_k), positions(h.positions),
        param_row(param_row), seed_lo(h.seed_lo), seed_hi(h.seed_hi) {
    view.temperature = temperature.get();
    view.top_k = top_k.get();
    view.top_p = top_p.get();
    view.seed_lo = seed_lo.get();
    view.seed_hi = seed_hi.get();
    view.positions = positions.get();
    view.param_row = param_row.empty() ? nullptr : this->param_row.get();
  }
};

HostParams uniform_params(int rows, float t, int k, float p, uint32_t seed) {
  HostParams h;
  for (int r = 0; r < rows; ++r) {
    h.temperature.push_back(t);
    h.top_k.push_back(k);
    h.top_p.push_back(p);
    h.seed_lo.push_back(seed + r);
    h.seed_hi.push_back(0);
    h.positions.push_back(r);
  }
  return h;
}

// Exact warped distribution (HF order: temperature -> top-k -> top-p).
std::vector<double> warped(const std::vector<float>& l, float t, int k, float p) {
  const int V = static_cast<int>(l.size());
  std::vector<int> idx(V);
  for (int i = 0; i < V; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
  std::vector<double> prob(V, 0.0);
  const double mx = l[idx[0]];
  int keep = (k > 0 && k < V) ? k : V;
  while (keep < V && l[idx[keep]] == l[idx[keep - 1]]) ++keep;  // ties at the k-th value are kept
  double total = 0;
  for (int i = 0; i < keep; ++i) total += std::exp((l[idx[i]] - mx) / t);
  double cum = 0;
  int keep_p = keep;
  if (p < 1.f) {
    for (int i = 0; i < keep; ++i) {
      cum += std::exp((l[idx[i]] - mx) / t) / total;
      if (cum >= p) {
        keep_p = i + 1;
        break;
      }
    }
  }
  double z = 0;
  for (int i = 0; i < keep_p; ++i) z += std::exp((l[idx[i]] - mx) / t);
  for (int i = 0; i < keep_p; ++i) prob[idx[i]] = std::exp((l[idx[i]] - mx) / t) / z;
  return prob;
}

}  // namespace

TEST(sampling_greedy_and_logprobs) {
  const int rows = 3, V = 1000;
  std::vector<float> logits = randn(rows * V, 2.f, 1);
  logits[0 * V + 17] = 50.f;
  logits[1 * V + 999] = 50.f;
  logits[2 * V + 5] = 50.f;
  logits[2 * V + 3] = 50.f;  // tie -> smaller index wins
  DeviceVec<float> dl(logits);
  DeviceParams p(uniform_params(rows, 0.f, 0, 1.f, 0));
  DeviceVec<int32_t> tok(rows);
  DeviceVec<float> lp(rows);
  kernels::sample_tokens(dl.get(), rows, V, p.view, kRngSample, tok.get(), lp.get(), nullptr, 0);
  auto t = tok.download();
  EXPECT_EQ(t[0], 17);
  EXPECT_EQ(t[1], 999);
  EXPECT_EQ(t[2], 3);
  auto l = lp.download();
  for (int r = 0; r < rows; ++r) {
    double mx = -1e30, s = 0;
    for (int i = 0; i < V; ++i) mx = std::max(mx, static_cast<double>(logits[r * V + i]));
    for (int i = 0; i < V; ++i) s += std::exp(logits[r * V + i] - mx);
    EXPECT_NEAR(l[r], logits[r * V + t[r]] - (mx + std::log(s)), 1e-4);
  }

  // gather_logprobs agrees with sample_tokens' log-probs.
  DeviceVec<float> g(rows);
  kernels::gather_logprobs(dl.get(), tok.get(), rows, V, g.get(), 0);
  auto gl = g.download();
  for (int r = 0; r < rows; ++r) EXPECT_NEAR(gl[r], l[r], 1e-5);
}

TEST(sampling_distribution_matches_warped) {
  struct Cfg {
    float t;
    int k;
    float p;
  };
  const int V = 50, rows = 6000;
  std::vector<float> base = randn(V, 1.5f, 7);
  for (Cfg c : {Cfg{1.f, 0, 1.f}, Cfg{0.7f, 10, 1.f}, Cfg{1.3f, 0, 0.8f}, Cfg{0.8f, 20, 0.9f}}) {
    std::vector<float> logits;
    for (int r = 0; r < rows; ++r) logits.insert(logits.end(), base.begin(), base.end());
    DeviceVec<float> dl(logits);
    DeviceParams p(uniform_params(rows, c.t, c.k, c.p, 1234));
    DeviceVec<int32_t> tok(rows);
    kernels::sample_tokens(dl.get(), rows, V, p.view, kRngSample, tok.get(), nullptr, nullptr, 0);
    auto t = tok.download();
    const auto expect = warped(base, c.t, c.k, c.p);
    std::vector<double> freq(V, 0);
    for (int x : t) freq[x] += 1.0 / rows;
    for (int i = 0; i < V; ++i) {
      if (expect[i] == 0) EXPECT_EQ(freq[i], 0.0);  // never sample outside the kept set
      EXPECT_NEAR(freq[i], expect[i], 0.025);
    }
  }
}

TEST(sampling_deterministic_and_large_vocab) {
  const int rows = 4, V = 128256;
  std::vector<float> logits = randn(rows * V, 3.f, 9);
  DeviceVec<float> dl(logits);
  DeviceParams p(uniform_params(rows, 0.9f, 50, 0.9f, 77));
  DeviceVec<int32_t> a(rows), b(rows);
  DeviceVec<RowStats> stats(rows);
  kernels::sample_tokens(dl.get(), rows, V, p.view, kRngSample, a.get(), nullptr, stats.get(), 0);
  kernels::sample_tokens(dl.get(), rows, V, p.view, kRngSample, b.get(), nullptr, nullptr, 0);
  auto ta = a.download(), tb = b.download();
  auto st = stats.download();
  for (int r = 0; r < rows; ++r) {
    EXPECT_EQ(ta[r], tb[r]);
    // Sampled token is within the top-50 of its row.
    std::vector<float> row(logits.begin() + static_cast<size_t>(r) * V, logits.begin() + static_cast<size_t>(r + 1) * V);
    std::vector<float> sorted = row;
    std::nth_element(sorted.begin(), sorted.begin() + 49, sorted.end(), std::greater<float>());
    EXPECT_TRUE(row[ta[r]] >= sorted[49]);
    EXPECT_TRUE(st[r].kept_mass > 0.f);
  }
}

TEST(speculative_verify_greedy) {
  // 2 sequences, K=3 drafts, vocab 16. Target argmax per row is known.
  const int S = 2, K = 3, V = 16, stride = 4;
  std::vector<float> target(S * (K + 1) * V, 0.f), draft(K * stride * V, 0.f);
  // seq 0: target wants 5, 6, 7, then bonus 8; drafts 5, 6, 9 -> accept 2, correct to 7.
  // seq 1: target wants 1, 2, 3, bonus 4; drafts 1, 2, 3 -> accept all, bonus 4.
  const int want[S][K + 1] = {{5, 6, 7, 8}, {1, 2, 3, 4}};
  const int drafts[S][K] = {{5, 6, 9}, {1, 2, 3}};
  std::vector<int32_t> dtok(K * stride, 0);
  for (int s = 0; s < S; ++s) {
    for (int i = 0; i <= K; ++i) target[(s * (K + 1) + i) * V + want[s][i]] = 10.f;
    for (int i = 0; i < K; ++i) dtok[i * stride + s] = drafts[s][i];
  }
  DeviceVec<float> dt(target), dd(draft);
  HostParams hp = uniform_params(S * (K + 1), 0.f, 0, 1.f, 0);
  std::vector<int32_t> param_row;
  for (int s = 0; s < S; ++s)
    for (int i = 0; i <= K; ++i) param_row.push_back(s);
  DeviceParams p(hp, param_row);
  DeviceVec<RowStats> ts(S * (K + 1)), ds(K * stride);
  kernels::compute_row_stats(dt.get(), S * (K + 1), V, p.view, ts.get(), 0);
  DeviceVec<int32_t> dtk(dtok), out(S * (K + 1)), cnt(S);
  DeviceVec<float> lp(S * (K + 1));
  kernels::speculative_verify(dt.get(), ts.get(), dd.get(), ds.get(), dtk.get(), S, K, stride, V, p.view, out.get(),
                              lp.get(), cnt.get(), 0);
  auto o = out.download();
  auto c = cnt.download();
  EXPECT_EQ(c[0], 3);
  EXPECT_EQ(o[0], 5);
  EXPECT_EQ(o[1], 6);
  EXPECT_EQ(o[2], 7);
  EXPECT_EQ(c[1], 4);
  for (int i = 0; i <= K; ++i) EXPECT_EQ(o[(K + 1) + i], want[1][i]);
}

TEST(speculative_verify_is_lossless) {
  // K=1: the first emitted token must follow the target distribution exactly,
  // whatever the draft distribution is.
  const int S = 20000, K = 1, V = 8;
  const std::vector<float> tl = {2.0f, 1.0f, 0.5f, 0.0f, -1.0f, 1.5f, -0.5f, 0.2f};
  const std::vector<float> dl = {0.0f, 2.0f, 0.5f, 1.0f, 0.0f, -1.0f, 0.3f, 0.0f};
  std::vector<float> target, draft;
  for (int s = 0; s < S; ++s)
    for (int i = 0; i <= K; ++i) target.insert(target.end(), tl.begin(), tl.end());
  for (int s = 0; s < S; ++s) draft.insert(draft.end(), dl.begin(), dl.end());
  DeviceVec<float> dt(target), dd(draft);
  // Draft rows: sample d_1 from the draft distribution with the draft stream.
  HostParams dh = uniform_params(S, 1.f, 0, 1.f, 99);
  DeviceParams dp(dh);
  DeviceVec<int32_t> dtok(S);
  DeviceVec<RowStats> dstats(S);
  kernels::sample_tokens(dd.get(), S, V, dp.view, kRngDraft, dtok.get(), nullptr, dstats.get(), 0);
  // Target rows (2 per sequence) share the sequence's params.
  HostParams th;
  std::vector<int32_t> param_row;
  for (int s = 0; s < S; ++s)
    for (int i = 0; i <= K; ++i) {
      th.temperature.push_back(1.f);
      th.top_k.push_back(0);
      th.top_p.push_back(1.f);
      th.seed_lo.push_back(99 + s);
      th.seed_hi.push_back(0);
      th.positions.push_back(s + i);
      param_row.push_back(static_cast<int32_t>(param_row.size()));
    }
  DeviceParams tp(th, param_row);
  DeviceVec<RowStats> tstats(S * (K + 1));
  kernels::compute_row_stats(dt.get(), S * (K + 1), V, tp.view, tstats.get(), 0);
  DeviceVec<int32_t> out(S * (K + 1)), cnt(S);
  DeviceVec<float> lp(S * (K + 1));
  kernels::speculative_verify(dt.get(), tstats.get(), dd.get(), dstats.get(), dtok.get(), S, K, S, V, tp.view,
                              out.get(), lp.get(), cnt.get(), 0);
  auto o = out.download();
  const auto expect = warped(tl, 1.f, 0, 1.f);
  std::vector<double> freq(V, 0);
  for (int s = 0; s < S; ++s) freq[o[s * (K + 1)]] += 1.0 / S;
  for (int i = 0; i < V; ++i) EXPECT_NEAR(freq[i], expect[i], 0.015);
}
