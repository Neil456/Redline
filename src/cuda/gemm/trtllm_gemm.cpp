// TensorRT-LLM weight-only GEMM backend (CutlassFpAIntBGemmRunner).
// Written against the TensorRT-LLM v0.16 C++ API; link libtensorrt_llm.so,
// which contains the kernel instantiations used here.
//
// Layout conversion (host, once at load time):
//   Redline INT4: uint8 [N, K/2], q in [0,15], w = q * s + o, s/o [N, K/g]
//   TRT-LLM:      signed int4 row-major [K, N] -> preprocess_weights_for_mixed_gemm,
//                 w = q_signed * s + z, s/z [K/g, N], so z = o + 8 s
//   Redline INT8: int8 [N, K], w = q * s[n]  -> row-major [K, N] -> preprocess.
#include <cuda_fp16.h>

#include <map>
#include <memory>
#include <vector>

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

#include "core/common.h"
#include "cuda/cuda_utils.h"
#include "cuda/gemm/backends.h"
#include "cuda/gemm/linear.h"

namespace redline {
namespace {

namespace tk = tensorrt_llm::kernels::cutlass_kernels;
namespace tkc = tensorrt_llm::cutlass_extensions;

using Int4Runner = tk::CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
                                                cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>;
using Int8Runner = tk::CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>;

int m_bucket(int m) {
  int b = 16;
  while (b < m) b *= 2;
  return b;
}

class TrtllmWeightOnlyGemmImpl final : public TrtllmWeightOnlyGemm {
 public:
  TrtllmWeightOnlyGemmImpl(MemoryLedger* ledger, int max_m) : ledger_(ledger), max_m_(max_m) {}

  void prepare(LinearWeight* w, const void* host_weight, const uint16_t* host_scales,
               const uint16_t* host_offsets, MemoryLedger* ledger) override {
    const int n = w->n, k = w->k;
    std::vector<int8_t> row_major;  // [K, N] signed, TensorRT-LLM input layout
    std::vector<uint16_t> scales, zeros;
    tk::QuantType qt;
    if (w->qtype == QuantType::kINT4) {
      qt = tk::QuantType::W4_A16;
      const int g = w->group_size, groups = k / g;
      const auto* src = static_cast<const uint8_t*>(host_weight);
      row_major.assign(static_cast<size_t>(k) * n / 2, 0);
      for (int r = 0; r < n; ++r) {
        for (int c = 0; c < k; ++c) {
          const uint8_t byte = src[static_cast<size_t>(r) * (k / 2) + c / 2];
          const int q = ((c & 1) ? (byte >> 4) : (byte & 0xF)) - 8;
          const size_t idx = static_cast<size_t>(c) * n + r;
          auto& out = reinterpret_cast<uint8_t&>(row_major[idx / 2]);
          out |= static_cast<uint8_t>((q & 0xF) << ((idx & 1) ? 4 : 0));
        }
      }
      scales.resize(static_cast<size_t>(groups) * n);
      zeros.resize(static_cast<size_t>(groups) * n);
      for (int r = 0; r < n; ++r) {
        for (int gi = 0; gi < groups; ++gi) {
          const uint16_t s = host_scales[static_cast<size_t>(r) * groups + gi];
          const float o = half_bits_to_float(host_offsets[static_cast<size_t>(r) * groups + gi]);
          scales[static_cast<size_t>(gi) * n + r] = s;
          zeros[static_cast<size_t>(gi) * n + r] = float_to_half_bits(o + 8.f * half_bits_to_float(s));
        }
      }
    } else {
      qt = tk::QuantType::W8_A16;
      const auto* src = static_cast<const int8_t*>(host_weight);
      row_major.resize(static_cast<size_t>(k) * n);
      for (int r = 0; r < n; ++r)
        for (int c = 0; c < k; ++c) row_major[static_cast<size_t>(c) * n + r] = src[static_cast<size_t>(r) * k + c];
      scales.assign(host_scales, host_scales + n);
    }
    std::vector<int8_t> processed(row_major.size());
    tk::preprocess_weights_for_mixed_gemm(processed.data(), row_major.data(),
                                          {static_cast<size_t>(k), static_cast<size_t>(n)}, qt);

    Stored st;
    st.weight = ledger->alloc("weights", processed.size());
    st.scales = ledger->alloc("weights", scales.size() * 2);
    RL_CUDA_CHECK(cudaMemcpy(st.weight.get(), processed.data(), processed.size(), cudaMemcpyHostToDevice));
    RL_CUDA_CHECK(cudaMemcpy(st.scales.get(), scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
    if (!zeros.empty()) {
      st.zeros = ledger->alloc("weights", zeros.size() * 2);
      RL_CUDA_CHECK(cudaMemcpy(st.zeros.get(), zeros.data(), zeros.size() * 2, cudaMemcpyHostToDevice));
    }
    w->trtllm_layout = true;
    w->weight = st.weight.get();
    w->scales = st.scales.as<half>();
    w->trt_zeros = st.zeros ? st.zeros.as<half>() : nullptr;
    w->offsets = nullptr;
    stored_.push_back(std::move(st));

    // Workspace sized for the largest shape served.
    const size_t ws = runner(w->qtype)->getWorkspaceSize(max_m_, n, k);
    if (ws > workspace_.bytes()) workspace_ = ledger_->alloc("workspace", ws);
  }

  void run(const LinearWeight& w, const half* x, int m, half* y, cudaStream_t stream) override {
    tk::CutlassFpAIntBGemmRunnerInterface* r = runner(w.qtype);
    const tkc::CutlassGemmConfig cfg = best_config(w, m, stream);
    gemm(r, w, x, m, y, cfg, stream);
  }

 private:
  struct Stored {
    DeviceBuffer weight, scales, zeros;
  };

  tk::CutlassFpAIntBGemmRunnerInterface* runner(QuantType q) {
    if (q == QuantType::kINT4) {
      if (!int4_) int4_ = std::make_unique<Int4Runner>();
      return int4_.get();
    }
    if (!int8_) int8_ = std::make_unique<Int8Runner>();
    return int8_.get();
  }

  void gemm(tk::CutlassFpAIntBGemmRunnerInterface* r, const LinearWeight& w, const half* x, int m,
            half* y, const tkc::CutlassGemmConfig& cfg, cudaStream_t stream) {
    const int group = w.qtype == QuantType::kINT4 ? w.group_size : w.k;
    r->gemm(x, w.weight, w.scales, w.trt_zeros, w.bias, y, m, w.n, w.k, group, cfg,
            workspace_.as<char>(), workspace_.bytes(), stream);
  }

  // Profiles every candidate tile config once per (m bucket, n, k) and caches
  // the fastest. The engine warms up eagerly before graph capture, so this
  // never runs inside a capture.
  tkc::CutlassGemmConfig best_config(const LinearWeight& w, int m, cudaStream_t stream) {
    const auto key = std::make_tuple(static_cast<int>(w.qtype), m_bucket(m), w.n, w.k);
    auto it = configs_.find(key);
    if (it != configs_.end()) return it->second;
    cudaStreamCaptureStatus capturing;
    RL_CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    RL_CHECK(capturing == cudaStreamCaptureStatusNone,
             "TensorRT-LLM GEMM config for ", w.name, " was not profiled before graph capture");
    tk::CutlassFpAIntBGemmRunnerInterface* r = runner(w.qtype);
    DeviceBuffer x(static_cast<size_t>(m) * w.k * sizeof(half));
    DeviceBuffer y(static_cast<size_t>(m) * w.n * sizeof(half));
    RL_CUDA_CHECK(cudaMemsetAsync(x.get(), 0, x.bytes(), stream));
    CudaTimer timer;
    float best_ms = 1e30f;
    tkc::CutlassGemmConfig best{};
    bool found = false;
    for (const auto& cfg : r->getConfigs()) {
      try {
        gemm(r, w, x.as<half>(), m, y.as<half>(), cfg, stream);  // warm-up / validity check
        timer.start(stream);
        for (int i = 0; i < 3; ++i) gemm(r, w, x.as<half>(), m, y.as<half>(), cfg, stream);
        timer.stop(stream);
        const float ms = timer.elapsed_ms();
        if (ms < best_ms) {
          best_ms = ms;
          best = cfg;
          found = true;
        }
      } catch (const std::exception&) {
        cudaGetLastError();  // config not supported for this shape
      }
    }
    RL_CHECK(found, "no TensorRT-LLM GEMM config works for ", w.name, " m=", m);
    configs_[key] = best;
    return best;
  }

  MemoryLedger* ledger_;
  int max_m_;
  std::unique_ptr<Int4Runner> int4_;
  std::unique_ptr<Int8Runner> int8_;
  std::vector<Stored> stored_;
  DeviceBuffer workspace_;
  std::map<std::tuple<int, int, int, int>, tkc::CutlassGemmConfig> configs_;
};

}  // namespace

std::unique_ptr<TrtllmWeightOnlyGemm> make_trtllm_gemm(MemoryLedger* ledger, int max_m) {
  return std::make_unique<TrtllmWeightOnlyGemmImpl>(ledger, max_m);
}

}  // namespace redline
