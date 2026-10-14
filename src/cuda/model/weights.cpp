#include "cuda/model/weights.h"

#include <cstring>
#include <map>

#include "core/int4_layout.h"
#include "core/safetensors.h"
#include "cpu/reference_model.h"
#include "cuda/cuda_utils.h"
#include "cuda/nvtx.h"

namespace redline {
namespace {

constexpr size_t kAlign = 256;
constexpr size_t kStagingBytes = 64u << 20;

// Streams host tensors into device memory through two pinned staging buffers
// so the memcpy into staging overlaps the previous chunk's DMA.
class PinnedUploader {
 public:
  PinnedUploader() : staging_{PinnedBuffer(kStagingBytes), PinnedBuffer(kStagingBytes)} {
    RL_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    for (auto& e : done_) RL_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }
  ~PinnedUploader() {
    cudaStreamSynchronize(stream_);
    for (auto& e : done_) cudaEventDestroy(e);
    cudaStreamDestroy(stream_);
  }

  void upload(void* dst, const void* src, size_t bytes) {
    const auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    while (bytes > 0) {
      const size_t n = std::min(bytes, kStagingBytes);
      RL_CUDA_CHECK(cudaEventSynchronize(done_[cur_]));  // staging buffer free again
      std::memcpy(staging_[cur_].get(), s, n);
      RL_CUDA_CHECK(cudaMemcpyAsync(d, staging_[cur_].get(), n, cudaMemcpyHostToDevice, stream_));
      RL_CUDA_CHECK(cudaEventRecord(done_[cur_], stream_));
      cur_ ^= 1;
      s += n;
      d += n;
      bytes -= n;
    }
  }

  void finish() { RL_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

 private:
  PinnedBuffer staging_[2];
  cudaEvent_t done_[2]{};
  cudaStream_t stream_{};
  int cur_ = 0;
};

}  // namespace

std::unique_ptr<ModelWeights> ModelWeights::load(const std::string& path, MemoryLedger* ledger,
                                                 TrtllmWeightOnlyGemm* trtllm,
                                                 bool int4_tensor_cores) {
  RL_NVTX_RANGE("load_weights", kNvtxHost);
  const double t0 = now_seconds();
  SafetensorsFile file(path);
  auto w = std::unique_ptr<ModelWeights>(new ModelWeights());
  w->config_ = read_model_config(file);
  const ModelConfig& cfg = w->config_;

  // Tensors that go to the device in Redline layout (quantized linears handled
  // by the TensorRT-LLM backend are converted separately).
  // The lm_head stays in Redline layout: it needs fp32 logits, which the
  // TensorRT-LLM runner does not produce.
  auto converted_by_trtllm = [&](const std::string& name) {
    if (trtllm == nullptr || cfg.weight_quant == QuantType::kFP16) return false;
    const bool linear = name.find(".wqkv.") != std::string::npos ||
                        name.find(".wo.") != std::string::npos ||
                        name.find(".w_gate_up.") != std::string::npos ||
                        name.find(".w_down.") != std::string::npos;
    return linear && name.find(".bias") == std::string::npos;
  };

  // INT4 linears that get the tensor-core layout: prefix -> interleave (gate, up) columns.
  struct TcLinear {
    int n, k;
    bool interleave;
  };
  std::map<std::string, TcLinear> tc;
  if (int4_tensor_cores) {
    auto consider = [&](const std::string& prefix, bool interleave) {
      if (!file.has(prefix + ".weight") || converted_by_trtllm(prefix + ".weight")) return;
      const TensorInfo& ti = file.info(prefix + ".weight");
      if (ti.dtype != DType::kU8 || ti.shape.size() != 2) return;
      const int n = static_cast<int>(ti.shape[0]), k = static_cast<int>(ti.shape[1] * 2);
      if (!int4_tc_supported(n, k, cfg.group_size)) return;
      // The interleaved layout permutes output columns, which a bias would have to follow.
      if (interleave && file.has(prefix + ".bias")) interleave = false;
      tc[prefix] = TcLinear{n, k, interleave};
    };
    for (int l = 0; l < cfg.num_layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      if (cfg.weight_quant == QuantType::kINT4) {
        consider(p + "wqkv", false);
        consider(p + "wo", false);
        consider(p + "w_gate_up", true);
        consider(p + "w_down", false);
      }
    }
    if (cfg.lm_head_quant == QuantType::kINT4) consider("lm_head", false);
  }
  auto tc_prefix = [&](const std::string& name) -> const TcLinear* {
    const size_t dot = name.rfind('.');
    auto it = tc.find(name.substr(0, dot));
    return it == tc.end() ? nullptr : &it->second;
  };

  std::map<std::string, size_t> offsets;
  size_t total = 0;
  for (const auto& name : file.names()) {
    if (converted_by_trtllm(name)) continue;
    offsets[name] = total;
    total += round_up(static_cast<int64_t>(file.info(name).nbytes), kAlign);
  }
  w->storage_ = ledger->alloc("weights", total);
  auto* base = static_cast<uint8_t*>(w->storage_.get());
  {
    PinnedUploader up;
    std::vector<uint8_t> tmp;
    for (const auto& kv : offsets) {
      const TensorInfo& ti = file.info(kv.first);
      const TcLinear* t = tc_prefix(kv.first);
      const std::string suffix = kv.first.substr(kv.first.rfind('.') + 1);
      if (t != nullptr && suffix != "bias") {
        tmp.resize(ti.nbytes);
        if (suffix == "weight") {
          relayout_int4_weight_tc(static_cast<const uint8_t*>(file.data(ti)), t->n, t->k,
                                  t->interleave, tmp.data());
        } else {  // scales / offsets
          relayout_int4_params_tc(static_cast<const uint16_t*>(file.data(ti)), t->n, t->k,
                                  cfg.group_size, t->interleave,
                                  reinterpret_cast<uint16_t*>(tmp.data()));
        }
        up.upload(base + kv.second, tmp.data(), ti.nbytes);
        up.finish();  // tmp is reused for the next tensor
      } else {
        up.upload(base + kv.second, file.data(ti), ti.nbytes);
      }
    }
    up.finish();
  }

  auto dev = [&](const std::string& name) -> const void* {
    auto it = offsets.find(name);
    return it == offsets.end() ? nullptr : base + it->second;
  };
  auto f16 = [&](const std::string& name, std::vector<int64_t> shape) {
    const TensorInfo& ti = file.info(name);
    RL_CHECK(ti.dtype == DType::kF16 && ti.shape == shape, "tensor ", name, " has unexpected ",
             dtype_name(ti.dtype), ti.shape_str());
    return static_cast<const half*>(dev(name));
  };
  auto linear = [&](const std::string& prefix, QuantType q, int n, int k, bool allow_trtllm = true) {
    LinearWeight lw;
    lw.name = prefix;
    lw.qtype = q;
    lw.n = n;
    lw.k = k;
    lw.group_size = cfg.group_size;
    const TensorInfo& wi = file.info(prefix + ".weight");
    if (q == QuantType::kFP16) {
      RL_CHECK(wi.dtype == DType::kF16 && wi.shape == std::vector<int64_t>({n, k}), prefix,
               ".weight: expected F16 [", n, ", ", k, "], got ", wi.shape_str());
    } else if (q == QuantType::kINT8) {
      RL_CHECK(wi.dtype == DType::kI8 && wi.shape == std::vector<int64_t>({n, k}), prefix,
               ".weight: expected I8 [", n, ", ", k, "]");
    } else {
      RL_CHECK(wi.dtype == DType::kU8 && wi.shape == std::vector<int64_t>({n, k / 2}), prefix,
               ".weight: expected U8 [", n, ", ", k / 2, "]");
    }
    if (file.has(prefix + ".bias")) lw.bias = f16(prefix + ".bias", {n});
    if (q != QuantType::kFP16 && trtllm != nullptr && allow_trtllm) {
      const void* off = file.has(prefix + ".offsets") ? file.data(prefix + ".offsets") : nullptr;
      trtllm->prepare(&lw, file.data(wi), static_cast<const uint16_t*>(file.data(prefix + ".scales")),
                      static_cast<const uint16_t*>(off), ledger);
      return lw;
    }
    lw.weight = dev(prefix + ".weight");
    if (q != QuantType::kFP16) lw.scales = static_cast<const half*>(dev(prefix + ".scales"));
    if (q == QuantType::kINT4) lw.offsets = static_cast<const half*>(dev(prefix + ".offsets"));
    if (auto it = tc.find(prefix); it != tc.end()) {
      lw.int4_tc = true;
      lw.interleaved = it->second.interleave;
    }
    return lw;
  };

  const int H = cfg.hidden_size;
  w->tok_embeddings = f16("tok_embeddings.weight", {cfg.vocab_size, H});
  for (int l = 0; l < cfg.num_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    LayerWeights lw;
    lw.attn_norm = f16(p + "attn_norm.weight", {H});
    lw.wqkv = linear(p + "wqkv", cfg.weight_quant, cfg.qkv_size(), H);
    lw.wo = linear(p + "wo", cfg.weight_quant, H, cfg.q_size());
    lw.ffn_norm = f16(p + "ffn_norm.weight", {H});
    lw.w_gate_up = linear(p + "w_gate_up", cfg.weight_quant, 2 * cfg.intermediate_size, H);
    lw.w_down = linear(p + "w_down", cfg.weight_quant, H, cfg.intermediate_size);
    w->layers.push_back(std::move(lw));
  }
  w->norm = f16("norm.weight", {H});
  if (file.has("lm_head.weight")) {
    w->lm_head = linear("lm_head", cfg.lm_head_quant, cfg.vocab_size, H, /*allow_trtllm=*/false);
  } else {
    RL_CHECK(cfg.tie_word_embeddings, "model has no lm_head.weight and untied embeddings");
    w->lm_head.name = "lm_head(tied)";
    w->lm_head.qtype = QuantType::kFP16;
    w->lm_head.n = cfg.vocab_size;
    w->lm_head.k = H;
    w->lm_head.weight = w->tok_embeddings;
  }
  RL_INFO("loaded ", path, " (", cfg.summary(), "): ", format_bytes(total), " in ",
          static_cast<int>((now_seconds() - t0) * 1000), " ms",
          tc.empty() ? "" : str_cat("; ", tc.size(), " INT4 linears in tensor-core layout"));
  return w;
}

}  // namespace redline
