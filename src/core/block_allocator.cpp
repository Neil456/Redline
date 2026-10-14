#include "core/block_allocator.h"

#include "core/common.h"

namespace redline {

BlockAllocator::BlockAllocator(int num_blocks, int num_reserved)
    : num_blocks_(num_blocks), num_reserved_(num_reserved), in_use_(num_blocks, 0) {
  RL_CHECK(num_blocks > num_reserved && num_reserved >= 1, "KV pool needs at least ",
           num_reserved + 1, " blocks, got ", num_blocks);
  free_list_.reserve(num_blocks);
  // Push in reverse so low block ids are handed out first.
  for (int b = num_blocks - 1; b >= num_reserved; --b) free_list_.push_back(b);
  for (int b = 0; b < num_reserved; ++b) in_use_[b] = 1;
}

int BlockAllocator::allocate() {
  if (free_list_.empty()) return -1;
  const int b = free_list_.back();
  free_list_.pop_back();
  in_use_[b] = 1;
  return b;
}

void BlockAllocator::free(int block) {
  RL_CHECK(block >= num_reserved_ && block < num_blocks_, "invalid block id ", block);
  RL_CHECK(in_use_[block], "double free of KV block ", block);
  in_use_[block] = 0;
  free_list_.push_back(block);
}

void BlockAllocator::free(const std::vector<int32_t>& blocks) {
  // Free in reverse so the next allocations reuse them in table order.
  for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) free(*it);
}

}  // namespace redline
