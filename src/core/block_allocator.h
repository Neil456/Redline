// Host-side allocator for fixed-size KV-cache blocks. The GPU KV pool is a
// flat array of blocks; sequences own an ordered list of block ids (their
// block table). Block 0 is reserved as a scratch block that padded batch rows
// (e.g. CUDA-graph padding) write into.
#pragma once

#include <cstdint>
#include <vector>

namespace redline {

class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks, int num_reserved = 1);

  int num_total() const { return num_blocks_; }
  int num_free() const { return static_cast<int>(free_list_.size()); }
  int num_used() const { return num_blocks_ - num_reserved_ - num_free(); }
  int scratch_block() const { return 0; }
  bool can_allocate(int n) const { return n <= num_free(); }

  // Returns -1 when the pool is exhausted.
  int allocate();
  void free(int block);
  void free(const std::vector<int32_t>& blocks);

 private:
  int num_blocks_;
  int num_reserved_;
  std::vector<int32_t> free_list_;  // LIFO: recently freed blocks are reused first (L2 locality)
  std::vector<uint8_t> in_use_;
};

}  // namespace redline
