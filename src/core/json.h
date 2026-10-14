// Minimal JSON value with parser and serializer. Used for model metadata,
// workload files and benchmark reports, so the runtime has no third-party
// dependencies beyond the CUDA toolkit.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace redline {

class Json {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  using Object = std::vector<std::pair<std::string, Json>>;
  using Array = std::vector<Json>;

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : type_(Type::kBool), bool_(b) {}
  Json(int v) : type_(Type::kNumber), num_(v) {}
  Json(int64_t v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Json(uint64_t v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Json(double v) : type_(Type::kNumber), num_(v) {}
  Json(float v) : type_(Type::kNumber), num_(v) {}
  Json(const char* s) : type_(Type::kString), str_(s) {}
  Json(std::string s) : type_(Type::kString), str_(std::move(s)) {}

  static Json object() { Json j; j.type_ = Type::kObject; return j; }
  static Json array() { Json j; j.type_ = Type::kArray; return j; }
  template <typename T>
  static Json array_of(const std::vector<T>& values) {
    Json j = array();
    for (const auto& v : values) j.push_back(Json(v));
    return j;
  }

  static Json parse(const std::string& text);
  static Json parse_file(const std::string& path);
  std::string dump(int indent = -1) const;
  void dump_file(const std::string& path, int indent = 2) const;

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::kNull; }
  bool is_bool() const { return type_ == Type::kBool; }
  bool is_number() const { return type_ == Type::kNumber; }
  bool is_string() const { return type_ == Type::kString; }
  bool is_array() const { return type_ == Type::kArray; }
  bool is_object() const { return type_ == Type::kObject; }

  bool as_bool() const;
  double as_double() const;
  int64_t as_int() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  size_t size() const;
  const Json& operator[](size_t i) const;
  // Object lookup; throws if the key is missing.
  const Json& operator[](const std::string& key) const;
  const Json* find(const std::string& key) const;
  bool contains(const std::string& key) const { return find(key) != nullptr; }

  // Typed lookups with defaults (missing key or null -> default).
  int64_t get_int(const std::string& key, int64_t def) const;
  double get_double(const std::string& key, double def) const;
  bool get_bool(const std::string& key, bool def) const;
  std::string get_string(const std::string& key, const std::string& def) const;

  // Mutation.
  Json& set(const std::string& key, Json value);
  Json& push_back(Json value);

 private:
  void dump_impl(std::string& out, int indent, int depth) const;

  Type type_ = Type::kNull;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  Array arr_;
  Object obj_;
};

}  // namespace redline
