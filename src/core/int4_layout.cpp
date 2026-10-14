#include "core/int4_layout.h"

#include <algorithm>
#include <thread>
#include <vector>

#include "core/common.h"

namespace redline {

bool int4_tc_supported(int n, int k, int group_size) {
  return n > 0 && k > 0 && n % 64 == 0 && k % 16 == 0 && group_size >= 16 && group_size % 16 == 0 &&
         k % group_size == 0;
}

namespace {

inline int src_nibble(const uint8_t* packed, int k, int row, int kk) {
  const uint8_t b = packed[static_cast<size_t>(row) * (k / 2) + kk / 2];
  return (kk & 1) ? (b >> 4) : (b & 0xF);
}

template <typename F>
void parallel_for(int n, F&& fn) {
  const int threads = std::max(1, std::min<int>(n, static_cast<int>(std::thread::hardware_concurrency())));
  if (threads <= 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> pool;
  const int chunk = static_cast<int>(ceil_div(n, threads));
  for (int t = 0; t < threads; ++t) {
    const int b = t * chunk, e = std::min(n, b + chunk);
    if (b < e) pool.emplace_back([&fn, b, e] { fn(b, e); });
  }
  for (auto& th : pool) th.join();
}

}  // namespace

void relayout_int4_weight_tc(const uint8_t* packed, int n, int k, bool interleave_halves,
                             uint8_t* dst) {
  RL_CHECK(n % 64 == 0 && k % 16 == 0, "tensor-core INT4 layout needs n % 64 == 0 and k % 16 == 0");
  const int ksteps = k / 16;
  parallel_for(n / 64, [&](int nt_begin, int nt_end) {
    for (int nt = nt_begin; nt < nt_end; ++nt) {
      for (int ks = 0; ks < ksteps; ++ks) {
        uint8_t* tile = dst + (static_cast<size_t>(nt) * ksteps + ks) * 512;
        for (int lane = 0; lane < 32; ++lane) {
          const int r = lane >> 2, t = lane & 3;
          for (int c = 0; c < 4; ++c) {
            uint32_t word = 0;
            for (int p = 0; p < 8; ++p) {
              const int j = 2 * c + ((p >> 1) & 1);
              const int col = nt * 64 + j * 8 + r;
              const int kk = ks * 16 + 2 * t + ((p >> 2) & 1) + 8 * (p & 1);
              const int row = int4_tc_source_row(col, n, interleave_halves);
              word |= static_cast<uint32_t>(src_nibble(packed, k, row, kk)) << (4 * p);
            }
            uint8_t* out = tile + lane * 16 + c * 4;  // little-endian word
            out[0] = static_cast<uint8_t>(word);
            out[1] = static_cast<uint8_t>(word >> 8);
            out[2] = static_cast<uint8_t>(word >> 16);
            out[3] = static_cast<uint8_t>(word >> 24);
          }
        }
      }
    }
  });
}

void relayout_int4_params_tc(const uint16_t* params, int n, int k, int group_size,
                             bool interleave_halves, uint16_t* dst) {
  RL_CHECK(int4_tc_supported(n, k, group_size), "unsupported tensor-core INT4 shape");
  const int groups = k / group_size;
  for (int nt = 0; nt < n / 64; ++nt) {
    for (int g = 0; g < groups; ++g) {
      uint16_t* out = dst + (static_cast<size_t>(nt) * groups + g) * 64;
      for (int r = 0; r < 8; ++r) {
        for (int j = 0; j < 8; ++j) {
          const int row = int4_tc_source_row(nt * 64 + j * 8 + r, n, interleave_halves);
          out[r * 8 + j] = params[static_cast<size_t>(row) * groups + g];
        }
      }
    }
  }
}

int int4_tc_get(const uint8_t* tc, int n, int k, int col, int kk) {
  (void)n;
  const int nt = col / 64, j = (col % 64) / 8, r = col % 8;
  const int ks = kk / 16, ko = kk % 16;
  const int t = (ko % 8) / 2;
  const int lane = r * 4 + t;
  const int c = j / 2;
  const int p = (j & 1) * 2 + (ko >= 8 ? 1 : 0) + (ko & 1) * 4;
  const uint8_t* word = tc + (static_cast<size_t>(nt) * (k / 16) + ks) * 512 + lane * 16 + c * 4;
  const uint8_t byte = word[p / 2];
  return (p & 1) ? (byte >> 4) : (byte & 0xF);
}

uint16_t int4_tc_param(const uint16_t* tc_params, int n, int k, int group_size, int col, int group) {
  (void)n;
  const int groups = k / group_size;
  const int nt = col / 64, j = (col % 64) / 8, r = col % 8;
  return tc_params[(static_cast<size_t>(nt) * groups + group) * 64 + r * 8 + j];
}

}  // namespace redline
