// Reader/writer for the safetensors container format, which Redline uses for
// its model files (.rlm). The reader memory-maps the file so tensors can be
// streamed to the GPU without an intermediate host copy.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace redline {

enum class DType { kF32, kF16, kBF16, kI8, kU8, kI32, kI64 };

size_t dtype_size(DType dt);
const char* dtype_name(DType dt);  // safetensors spelling, e.g. "F16"
DType dtype_from_name(const std::string& name);

struct TensorInfo {
  std::string name;
  DType dtype = DType::kF32;
  std::vector<int64_t> shape;
  uint64_t offset = 0;  // relative to the start of the data section
  uint64_t nbytes = 0;

  int64_t numel() const;
  std::string shape_str() const;
};

class SafetensorsFile {
 public:
  explicit SafetensorsFile(const std::string& path);
  ~SafetensorsFile();
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

  const std::string& path() const { return path_; }
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorInfo& info(const std::string& name) const;
  const void* data(const std::string& name) const;
  const void* data(const TensorInfo& info) const;
  std::vector<std::string> names() const;
  const std::map<std::string, std::string>& metadata() const { return metadata_; }
  uint64_t data_bytes() const { return data_size_; }

 private:
  std::string path_;
  int fd_ = -1;
  void* map_ = nullptr;
  size_t map_size_ = 0;
  const uint8_t* data_base_ = nullptr;
  uint64_t data_size_ = 0;
  std::map<std::string, TensorInfo> tensors_;
  std::map<std::string, std::string> metadata_;
};

// Accumulates tensors in memory and writes one safetensors file. Intended for
// tests and small tools; the Python exporter streams large models instead.
class SafetensorsWriter {
 public:
  void add(const std::string& name, DType dtype, const std::vector<int64_t>& shape,
           const void* data, size_t nbytes);
  void set_metadata(const std::string& key, const std::string& value);
  void write(const std::string& path) const;

 private:
  struct Entry {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
  };
  std::vector<Entry> entries_;
  std::map<std::string, std::string> metadata_;
};

}  // namespace redline
