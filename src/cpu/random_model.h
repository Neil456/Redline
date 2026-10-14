// Generates small random Llama models in the Redline file format, with the
// same quantization schemes as the Python exporter. Used by self-contained
// tests (CPU and CUDA) that should not depend on Python or downloaded weights.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/model_config.h"

namespace redline {

// Symmetric per-row INT8: q in [-127, 127], w ~= q * scale[row].
void quantize_int8_rows(const std::vector<float>& w, int n, int k, std::vector<int8_t>* q,
                        std::vector<uint16_t>* scales_f16);

// Asymmetric group-wise INT4: q in [0, 15], w ~= q * scale + offset per group.
// Two values per byte, the even column in the low nibble.
void quantize_int4_groups(const std::vector<float>& w, int n, int k, int group_size,
                          std::vector<uint8_t>* packed, std::vector<uint16_t>* scales_f16,
                          std::vector<uint16_t>* offsets_f16);

// Small config suitable for tests (GQA, head_dim 64).
ModelConfig tiny_test_config(QuantType quant = QuantType::kFP16);

void write_random_model(const std::string& path, const ModelConfig& cfg, uint64_t seed);

}  // namespace redline
