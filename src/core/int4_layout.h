// INT4 weight re-layout for the tensor-core W4A16 GEMM (host side).
//
// Model files store INT4 weights row-major: packed[n][k/2], the low nibble holding the even k,
// with fp16 scales / offsets [n][k/g] (w = q * s + o). The tensor-core kernel instead wants each
// thread's 16-byte load to be exactly its B fragments of mma.sync.m16n8k16, so the weights are
// re-laid out once at load time:
//
//   tile (nt, ks) = output columns [64 nt, 64 nt + 64) x k [16 ks, 16 ks + 16), 512 bytes;
//   tiles are ordered [nt][ks]; inside a tile, lane l owns bytes [16 l, 16 l + 16) = 4 words.
//   Word c of lane l holds the 8 nibbles of n8 tiles j = 2c and 2c + 1 for column
//   n = 64 nt + 8 j + l / 4 and k = 16 ks + 2 (l % 4) + {0, 1, 8, 9}, ordered so that
//   two lop3 masks (0x000f000f, 0x00f000f0) on the word and on word >> 8 yield the
//   fragment registers {b0, b1} and {b2, b3} directly:
//     nibble p -> j = 2c + ((p >> 1) & 1),  k offset = ((p >> 2) & 1) + 8 * (p & 1).
//
//   Scales / offsets: [nt][group][r = 0..7][j = 0..7] halves, so lane l reads the 8 values it
//   needs (column 64 nt + 8 j + l / 4 for j = 0..7) with one 16-byte load.
//
// With `interleave_halves` the output columns of a [gate; up] matrix (n = 2 I) are interleaved
// (column 2i = gate row i, 2i + 1 = up row I + i), which lets the GEMM epilogue apply SiLU(gate) *
// up in registers: each thread owns adjacent output column pairs.
#pragma once

#include <cstdint>

namespace redline {

bool int4_tc_supported(int n, int k, int group_size);

// Source row (in the model file) of tensor-core column `col`.
inline int int4_tc_source_row(int col, int n, bool interleave_halves) {
  if (!interleave_halves) return col;
  return (col & 1) ? n / 2 + col / 2 : col / 2;
}

// packed: uint8 [n][k/2] -> dst: same size, tensor-core order. Multi-threaded.
void relayout_int4_weight_tc(const uint8_t* packed, int n, int k, bool interleave_halves,
                             uint8_t* dst);
// fp16 bit patterns [n][k/g] -> [n/64][k/g][8][8].
void relayout_int4_params_tc(const uint16_t* params, int n, int k, int group_size,
                             bool interleave_halves, uint16_t* dst);

// Reads q (0..15) for tensor-core column `col` and input `kk` from a re-laid-out buffer (the
// inverse of the mapping above; used by tests).
int int4_tc_get(const uint8_t* tc, int n, int k, int col, int kk);
uint16_t int4_tc_param(const uint16_t* tc_params, int n, int k, int group_size, int col, int group);

}  // namespace redline
