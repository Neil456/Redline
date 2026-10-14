#include "cuda/device_memory.h"

#include <utility>

#include "cuda/cuda_utils.h"

namespace redline {

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
  if (bytes > 0) {
    const cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
      ptr_ = nullptr;
      throw Error(str_cat("cudaMalloc(", format_bytes(bytes), ") failed: ", cudaGetErrorString(err)));
    }
  }
}

DeviceBuffer::~DeviceBuffer() {
  if (ptr_ != nullptr) cudaFree(ptr_);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept
    : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
  if (this != &o) {
    if (ptr_ != nullptr) cudaFree(ptr_);
    ptr_ = std::exchange(o.ptr_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
  }
  return *this;
}

PinnedBuffer::PinnedBuffer(size_t bytes) : bytes_(bytes) {
  if (bytes > 0) RL_CUDA_CHECK(cudaHostAlloc(&ptr_, bytes, cudaHostAllocDefault));
}

PinnedBuffer::~PinnedBuffer() {
  if (ptr_ != nullptr) cudaFreeHost(ptr_);
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& o) noexcept
    : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& o) noexcept {
  if (this != &o) {
    if (ptr_ != nullptr) cudaFreeHost(ptr_);
    ptr_ = std::exchange(o.ptr_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
  }
  return *this;
}

size_t MemoryLedger::get(const std::string& category) const {
  auto it = bytes_.find(category);
  return it == bytes_.end() ? 0 : it->second;
}

size_t MemoryLedger::total() const {
  size_t t = 0;
  for (const auto& kv : bytes_) t += kv.second;
  return t;
}

Json MemoryLedger::to_json() const {
  Json j = Json::object();
  for (const auto& kv : bytes_) j.set(kv.first, static_cast<uint64_t>(kv.second));
  j.set("total", static_cast<uint64_t>(total()));
  return j;
}

DeviceBuffer MemoryLedger::alloc(const std::string& category, size_t bytes) {
  add(category, bytes);
  return DeviceBuffer(bytes);
}

DeviceMemoryInfo device_memory_info() {
  DeviceMemoryInfo info;
  RL_CUDA_CHECK(cudaMemGetInfo(&info.free, &info.total));
  return info;
}

}  // namespace redline
