#include "core/json.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "core/common.h"

namespace redline {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}

  Json parse_document() {
    Json v = parse_value();
    skip_ws();
    if (pos_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    throw Error(str_cat("JSON parse error at offset ", pos_, ": ", what));
  }

  void skip_ws() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) {
      ++pos_;
    }
  }

  char peek() {
    skip_ws();
    if (pos_ >= s_.size()) fail("unexpected end of input");
    return s_[pos_];
  }

  void expect(char c) {
    if (peek() != c) fail(str_cat("expected '", c, "'"));
    ++pos_;
  }

  bool consume_literal(const char* lit) {
    size_t n = std::char_traits<char>::length(lit);
    if (s_.compare(pos_, n, lit) == 0) {
      pos_ += n;
      return true;
    }
    return false;
  }

  Json parse_value() {
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return Json(parse_string());
    if (consume_literal("true")) return Json(true);
    if (consume_literal("false")) return Json(false);
    if (consume_literal("null")) return Json();
    if (consume_literal("NaN")) return Json(std::nan(""));
    if (consume_literal("Infinity")) return Json(HUGE_VAL);
    if (consume_literal("-Infinity")) return Json(-HUGE_VAL);
    return parse_number();
  }

  Json parse_object() {
    expect('{');
    Json obj = Json::object();
    if (peek() == '}') {
      ++pos_;
      return obj;
    }
    while (true) {
      if (peek() != '"') fail("expected object key");
      std::string key = parse_string();
      expect(':');
      obj.set(key, parse_value());
      char c = peek();
      ++pos_;
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}'");
    }
    return obj;
  }

  Json parse_array() {
    expect('[');
    Json arr = Json::array();
    if (peek() == ']') {
      ++pos_;
      return arr;
    }
    while (true) {
      arr.push_back(parse_value());
      char c = peek();
      ++pos_;
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']'");
    }
    return arr;
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  uint32_t parse_hex4() {
    if (pos_ + 4 > s_.size()) fail("truncated \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad hex digit");
    }
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) fail("unterminated string");
      char c = s_[pos_++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= s_.size()) fail("unterminated escape");
      char e = s_[pos_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = parse_hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 6 <= s_.size() && s_[pos_] == '\\' &&
              s_[pos_ + 1] == 'u') {
            pos_ += 2;
            uint32_t lo = parse_hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }

  Json parse_number() {
    size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
        ++pos_;
      } else {
        break;
      }
    }
    if (start == pos_) fail("unexpected character");
    const std::string tok = s_.substr(start, pos_ - start);
    char* end = nullptr;
    double v = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0') fail("bad number '" + tok + "'");
    return Json(v);
  }

  const std::string& s_;
  size_t pos_ = 0;
};

void escape_string(std::string& out, const std::string& s) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

void append_number(std::string& out, double v) {
  if (std::isnan(v) || std::isinf(v)) {
    out += "null";
    return;
  }
  if (v == std::floor(v) && std::fabs(v) < 9.0e15) {
    out += std::to_string(static_cast<int64_t>(v));
    return;
  }
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof(buf), v);
  out.append(buf, res.ptr);
}

}  // namespace

Json Json::parse(const std::string& text) { return Parser(text).parse_document(); }

Json Json::parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("cannot open JSON file: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse(ss.str());
}

void Json::dump_file(const std::string& path, int indent) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw Error("cannot write JSON file: " + path);
  out << dump(indent) << "\n";
}

std::string Json::dump(int indent) const {
  std::string out;
  dump_impl(out, indent, 0);
  return out;
}

void Json::dump_impl(std::string& out, int indent, int depth) const {
  auto newline = [&](int d) {
    if (indent < 0) return;
    out += '\n';
    out.append(static_cast<size_t>(indent * d), ' ');
  };
  switch (type_) {
    case Type::kNull: out += "null"; break;
    case Type::kBool: out += bool_ ? "true" : "false"; break;
    case Type::kNumber: append_number(out, num_); break;
    case Type::kString: escape_string(out, str_); break;
    case Type::kArray: {
      out += '[';
      for (size_t i = 0; i < arr_.size(); ++i) {
        if (i) out += ',';
        newline(depth + 1);
        arr_[i].dump_impl(out, indent, depth + 1);
      }
      if (!arr_.empty()) newline(depth);
      out += ']';
      break;
    }
    case Type::kObject: {
      out += '{';
      for (size_t i = 0; i < obj_.size(); ++i) {
        if (i) out += ',';
        newline(depth + 1);
        escape_string(out, obj_[i].first);
        out += indent < 0 ? ":" : ": ";
        obj_[i].second.dump_impl(out, indent, depth + 1);
      }
      if (!obj_.empty()) newline(depth);
      out += '}';
      break;
    }
  }
}

bool Json::as_bool() const {
  RL_CHECK(type_ == Type::kBool, "JSON value is not a bool");
  return bool_;
}
double Json::as_double() const {
  RL_CHECK(type_ == Type::kNumber, "JSON value is not a number");
  return num_;
}
int64_t Json::as_int() const {
  RL_CHECK(type_ == Type::kNumber, "JSON value is not a number");
  return static_cast<int64_t>(std::llround(num_));
}
const std::string& Json::as_string() const {
  RL_CHECK(type_ == Type::kString, "JSON value is not a string");
  return str_;
}
const Json::Array& Json::as_array() const {
  RL_CHECK(type_ == Type::kArray, "JSON value is not an array");
  return arr_;
}
const Json::Object& Json::as_object() const {
  RL_CHECK(type_ == Type::kObject, "JSON value is not an object");
  return obj_;
}

size_t Json::size() const {
  if (type_ == Type::kArray) return arr_.size();
  if (type_ == Type::kObject) return obj_.size();
  return 0;
}

const Json& Json::operator[](size_t i) const {
  RL_CHECK(type_ == Type::kArray && i < arr_.size(), "JSON array index out of range: ", i);
  return arr_[i];
}

const Json* Json::find(const std::string& key) const {
  if (type_ != Type::kObject) return nullptr;
  for (const auto& kv : obj_) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

const Json& Json::operator[](const std::string& key) const {
  const Json* v = find(key);
  RL_CHECK(v != nullptr, "missing JSON key '", key, "'");
  return *v;
}

int64_t Json::get_int(const std::string& key, int64_t def) const {
  const Json* v = find(key);
  return (v == nullptr || v->is_null()) ? def : v->as_int();
}
double Json::get_double(const std::string& key, double def) const {
  const Json* v = find(key);
  return (v == nullptr || v->is_null()) ? def : v->as_double();
}
bool Json::get_bool(const std::string& key, bool def) const {
  const Json* v = find(key);
  return (v == nullptr || v->is_null()) ? def : v->as_bool();
}
std::string Json::get_string(const std::string& key, const std::string& def) const {
  const Json* v = find(key);
  return (v == nullptr || v->is_null()) ? def : v->as_string();
}

Json& Json::set(const std::string& key, Json value) {
  if (type_ == Type::kNull) type_ = Type::kObject;
  RL_CHECK(type_ == Type::kObject, "JSON set() on non-object");
  for (auto& kv : obj_) {
    if (kv.first == key) {
      kv.second = std::move(value);
      return *this;
    }
  }
  obj_.emplace_back(key, std::move(value));
  return *this;
}

Json& Json::push_back(Json value) {
  if (type_ == Type::kNull) type_ = Type::kArray;
  RL_CHECK(type_ == Type::kArray, "JSON push_back() on non-array");
  arr_.push_back(std::move(value));
  return *this;
}

}  // namespace redline
