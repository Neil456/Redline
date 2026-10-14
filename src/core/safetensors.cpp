#include "core/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>

#include "core/common.h"
#include "core/json.h"

namespace redline {

size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::kF32: return 4;
    case DType::kF16: return 2;
    case DType::kBF16: return 2;
    case DType::kI8: return 1;
    case DType::kU8: return 1;
    case DType::kI32: return 4;
    case DType::kI64: return 8;
  }
  return 0;
}

const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::kF32: return "F32";
    case DType::kF16: return "F16";
    case DType::kBF16: return "BF16";
    case DType::kI8: return "I8";
    case DType::kU8: return "U8";
    case DType::kI32: return "I32";
    case DType::kI64: return "I64";
  }
  return "?";
}

DType dtype_from_name(const std::string& name) {
  if (name == "F32") return DType::kF32;
  if (name == "F16") return DType::kF16;
  if (name == "BF16") return DType::kBF16;
  if (name == "I8") return DType::kI8;
  if (name == "U8") return DType::kU8;
  if (name == "I32") return DType::kI32;
  if (name == "I64") return DType::kI64;
  throw Error("unsupported safetensors dtype: " + name);
}

int64_t TensorInfo::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

std::string TensorInfo::shape_str() const {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

SafetensorsFile::SafetensorsFile(const std::string& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw Error("cannot open model file: " + path);
  struct stat st;
  if (::fstat(fd_, &st) != 0) throw Error("cannot stat model file: " + path);
  map_size_ = static_cast<size_t>(st.st_size);
  RL_CHECK(map_size_ >= 8, "file too small to be safetensors: ", path);
  map_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (map_ == MAP_FAILED) {
    map_ = nullptr;
    throw Error("mmap failed for " + path);
  }
  const auto* base = static_cast<const uint8_t*>(map_);
  uint64_t header_len = 0;
  std::memcpy(&header_len, base, 8);  // little-endian on all supported hosts
  RL_CHECK(8 + header_len <= map_size_, "corrupt safetensors header length in ", path);
  const std::string header(reinterpret_cast<const char*>(base + 8), header_len);
  data_base_ = base + 8 + header_len;
  data_size_ = map_size_ - 8 - header_len;

  const Json root = Json::parse(header);
  for (const auto& kv : root.as_object()) {
    if (kv.first == "__metadata__") {
      for (const auto& m : kv.second.as_object()) metadata_[m.first] = m.second.as_string();
      continue;
    }
    TensorInfo ti;
    ti.name = kv.first;
    ti.dtype = dtype_from_name(kv.second["dtype"].as_string());
    for (const auto& d : kv.second["shape"].as_array()) ti.shape.push_back(d.as_int());
    const auto& offs = kv.second["data_offsets"];
    const uint64_t begin = static_cast<uint64_t>(offs[0].as_int());
    const uint64_t end = static_cast<uint64_t>(offs[1].as_int());
    RL_CHECK(end >= begin && end <= data_size_, "tensor ", ti.name, " out of bounds");
    ti.offset = begin;
    ti.nbytes = end - begin;
    RL_CHECK(ti.nbytes == static_cast<uint64_t>(ti.numel()) * dtype_size(ti.dtype),
             "tensor ", ti.name, " has inconsistent byte size");
    tensors_.emplace(ti.name, std::move(ti));
  }
}

SafetensorsFile::~SafetensorsFile() {
  if (map_ != nullptr) ::munmap(map_, map_size_);
  if (fd_ >= 0) ::close(fd_);
}

const TensorInfo& SafetensorsFile::info(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw Error("tensor not found in " + path_ + ": " + name);
  return it->second;
}

const void* SafetensorsFile::data(const std::string& name) const { return data(info(name)); }

const void* SafetensorsFile::data(const TensorInfo& ti) const { return data_base_ + ti.offset; }

std::vector<std::string> SafetensorsFile::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& kv : tensors_) out.push_back(kv.first);
  return out;
}

void SafetensorsWriter::add(const std::string& name, DType dtype,
                            const std::vector<int64_t>& shape, const void* data,
                            size_t nbytes) {
  int64_t numel = 1;
  for (int64_t d : shape) numel *= d;
  RL_CHECK(static_cast<size_t>(numel) * dtype_size(dtype) == nbytes, "size mismatch for ", name);
  Entry e{name, dtype, shape, {}};
  e.bytes.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + nbytes);
  entries_.push_back(std::move(e));
}

void SafetensorsWriter::set_metadata(const std::string& key, const std::string& value) {
  metadata_[key] = value;
}

void SafetensorsWriter::write(const std::string& path) const {
  Json header = Json::object();
  if (!metadata_.empty()) {
    Json meta = Json::object();
    for (const auto& kv : metadata_) meta.set(kv.first, kv.second);
    header.set("__metadata__", meta);
  }
  uint64_t offset = 0;
  for (const auto& e : entries_) {
    Json t = Json::object();
    t.set("dtype", dtype_name(e.dtype));
    t.set("shape", Json::array_of(e.shape));
    Json offs = Json::array();
    offs.push_back(Json(offset));
    offs.push_back(Json(offset + e.bytes.size()));
    t.set("data_offsets", offs);
    header.set(e.name, t);
    offset += e.bytes.size();
  }
  std::string h = header.dump();
  while (h.size() % 8 != 0) h += ' ';
  std::ofstream out(path, std::ios::binary);
  if (!out) throw Error("cannot write " + path);
  const uint64_t len = h.size();
  out.write(reinterpret_cast<const char*>(&len), 8);
  out.write(h.data(), static_cast<std::streamsize>(h.size()));
  for (const auto& e : entries_) {
    out.write(reinterpret_cast<const char*>(e.bytes.data()),
              static_cast<std::streamsize>(e.bytes.size()));
  }
  if (!out) throw Error("short write to " + path);
}

}  // namespace redline
