// GPU / pinned memory ownership and accounting. Every device buffer the
// runtime uses is allocated once at engine construction and reused for the
// lifetime of the engine; nothing is allocated on the decode path.
#pragma once

#include <cstddef>
#include <map>
#include <string>

#include "core/json.h"

namespace redline {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer&& o) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  template <typename T>
  T* as() const { return static_cast<T*>(ptr_); }
  size_t bytes() const { return bytes_; }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

// Page-locked host memory: required for truly asynchronous H2D/D2H copies.
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t bytes);
  ~PinnedBuffer();
  PinnedBuffer(PinnedBuffer&& o) noexcept;
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept;
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void* get() const { return ptr_; }
  template <typename T>
  T* as() const { return static_cast<T*>(ptr_); }
  size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

// Tracks bytes per category (weights, kv_cache, activations, workspace, ...).
class MemoryLedger {
 public:
  void add(const std::string& category, size_t bytes) { bytes_[category] += bytes; }
  size_t get(const std::string& category) const;
  size_t total() const;
  Json to_json() const;

  // Allocates a device buffer and records it under `category`.
  DeviceBuffer alloc(const std::string& category, size_t bytes);

 private:
  std::map<std::string, size_t> bytes_;
};

struct DeviceMemoryInfo {
  size_t free = 0;
  size_t total = 0;
  size_t used() const { return total - free; }
};
DeviceMemoryInfo device_memory_info();

}  // namespace redline
