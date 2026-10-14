// NVTX ranges for Nsight Systems timelines. Compiled out unless
// REDLINE_WITH_NVTX is defined; near-zero cost when no profiler is attached.
#pragma once

#include <cstdint>

#ifdef REDLINE_WITH_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace redline {

enum NvtxColor : uint32_t {
  kNvtxStep = 0xFF4C72B0,
  kNvtxPrefill = 0xFFDD8452,
  kNvtxDecode = 0xFF55A868,
  kNvtxSample = 0xFFC44E52,
  kNvtxHost = 0xFF8172B3,
  kNvtxCopy = 0xFF937860,
  kNvtxLayer = 0xFF64B5CD,
  kNvtxSpec = 0xFFDA8BC3,
};

class NvtxRange {
 public:
#ifdef REDLINE_WITH_NVTX
  explicit NvtxRange(const char* name, uint32_t color = kNvtxStep) {
    nvtxEventAttributes_t a = {};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.colorType = NVTX_COLOR_ARGB;
    a.color = color;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = name;
    nvtxRangePushEx(&a);
  }
  ~NvtxRange() { nvtxRangePop(); }
#else
  explicit NvtxRange(const char*, uint32_t = 0) {}
#endif
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

inline void nvtx_mark(const char* name) {
#ifdef REDLINE_WITH_NVTX
  nvtxMarkA(name);
#else
  (void)name;
#endif
}

}  // namespace redline

#define RL_NVTX_CONCAT2(a, b) a##b
#define RL_NVTX_CONCAT(a, b) RL_NVTX_CONCAT2(a, b)
#define RL_NVTX_RANGE(...) ::redline::NvtxRange RL_NVTX_CONCAT(rl_nvtx_, __LINE__)(__VA_ARGS__)
