/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pipeline/jit/pi/graph_guard/info.h"
#include <sstream>
#include <map>

namespace mindspore {
namespace pijit {

static constexpr char SEP_FLAG[] = "\\";
static constexpr char BEGIN_FLAG[] = "{";
static constexpr char END_FLAG[] = "}";
static constexpr char ARRAY_BEGIN_FLAG[] = "[";
static constexpr char ARRAY_END_FLAG[] = "[";

InfoPack::InfoPack() : hash_(INVALID_HASH), id_(INVALID_ID), info_("") {}

InfoPack::InfoPack(const InfoPack &dup) : hash_(dup.Hash()), id_(dup.Id()), info_(dup.ToString()) {}

size_t InfoPack::Hash() const { return hash_; }

size_t InfoPack::Id() const { return id_; }

std::string InfoPack::ToString() const { return info_; }

void InfoPack::Update() {
  if (hash_ == INVALID_HASH) {
    hash_ = CalcHash(info_);
  }
  if (id_ == INVALID_ID) {
    id_ = CalcId(info_);
  }
}

bool InfoPack::operator()(const InfoPack &lhs, const InfoPack &rhs) const { return lhs.ToString() == rhs.ToString(); }

size_t InfoPack::operator()(const InfoPack &k) const { return Hash(); }

InfoPack &InfoPack::Begin() {
  info_ += BEGIN_FLAG;
  return *this;
}

InfoPack &InfoPack::End() {
  if (info_.rfind(SEP_FLAG) == info_.size() - 1) {
    info_ = info_.substr(0, info_.size() - 1);
  }
  info_ += END_FLAG;
  return *this;
}

#define APPEND_SCALAR(v) info_ += std::to_string(v) + SEP_FLAG
#define APPEND_SCALAR_PACK(v) \
  std::stringstream ss;       \
  ss << std::hex << v;        \
  info_ += ss.str() + SEP_FLAG

InfoPack &InfoPack::operator<<(int8_t v) {
  APPEND_SCALAR(v);
  return *this;
}

InfoPack &InfoPack::operator<<(uint8_t v) {
  APPEND_SCALAR(v);
  return *this;
}

InfoPack &InfoPack::operator<<(int16_t v) {
  APPEND_SCALAR_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(uint16_t v) {
  APPEND_SCALAR_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(int32_t v) {
  APPEND_SCALAR_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(uint32_t v) {
  APPEND_SCALAR_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(int64_t v) {
  APPEND_SCALAR_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(uint64_t v) {
  APPEND_SCALAR_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(float v) {
  APPEND_SCALAR_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(double v) {
  APPEND_SCALAR(v);
  return *this;
}

InfoPack &InfoPack::operator<<(bool v) {
  info_ += std::to_string(v ? 1 : 0) + SEP_FLAG;
  return *this;
}

InfoPack &InfoPack::operator<<(void *v) {
  std::stringstream ss;
  ss << v;
  info_ += ss.str() + SEP_FLAG;
  return *this;
}

InfoPack &InfoPack::operator<<(PyObject *v) {
  info_ += std::to_string(v != nullptr ? 1 : 0) + SEP_FLAG;  
  if (v != nullptr) {
    info_ += std::to_string(CalcString(std::string(py::str(v)))) + SEP_FLAG;
  }
  return *this;
}

InfoPack &InfoPack::operator<<(mindspore::BasePtr v) {
  info_ += std::to_string(v != nullptr ? 1 : 0) + SEP_FLAG;
  if (v != nullptr) {
    info_ += v->ToString();
  }
  return *this;
}

InfoPack &InfoPack::operator<<(const std::string &v) {
  info_ += std::to_string(CalcString(v)) + SEP_FLAG;
  return *this;
}

#define APPEND_ARRAY(v)                            \
  info_ += ARRAY_BEGIN_FLAG;                       \
  for (auto val : v) {                             \
    info_ += std::to_string(val) + SEP_FLAG;       \
  }                                                \
  if (info_.rfind(SEP_FLAG) == info_.size() - 1) { \
    info_ = info_.substr(0, info_.size() - 1);     \
  }                                                \
  info_ += ARRAY_END_FLAG;                         \
  info_ += SEP_FLAG

#define APPEND_ARRAY_PACK(v)                       \
  info_ += ARRAY_BEGIN_FLAG;                       \
  for (auto val : v) {                             \
    std::stringstream ss;                          \
    ss << std::hex << val;                         \
    info_ += ss.str() + SEP_FLAG;                  \
  }                                                \
  if (info_.rfind(SEP_FLAG) == info_.size() - 1) { \
    info_ = info_.substr(0, info_.size() - 1);     \
  }                                                \
  info_ += ARRAY_END_FLAG;                         \
  info_ += SEP_FLAG

InfoPack &InfoPack::operator<<(const std::vector<int8_t> &v) {
  APPEND_ARRAY(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<uint8_t> &v) {
  APPEND_ARRAY(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<int16_t> &v) {
  APPEND_ARRAY_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<uint16_t> &v) {
  APPEND_ARRAY_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<int32_t> &v) {
  APPEND_ARRAY_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<uint32_t> &v) {
  APPEND_ARRAY_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<int64_t> &v) {
  APPEND_ARRAY_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<uint64_t> &v) {
  APPEND_ARRAY_PACK(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<float> &v) {
  APPEND_ARRAY(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<double> &v) {
  APPEND_ARRAY(v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<bool> &v) {
  info_ += ARRAY_BEGIN_FLAG;
  for (auto val : v) {
    info_ += std::to_string(val ? 1 : 0) + SEP_FLAG;
  }
  if (info_.rfind(SEP_FLAG) == info_.size() - 1) {
    info_ = info_.substr(0, info_.size() - 1);
  }
  info_ += ARRAY_END_FLAG;
  info_ += SEP_FLAG;
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<std::string> &v) {
  info_ += ARRAY_BEGIN_FLAG;
  for (auto val : v) {
    info_ += std::to_string(CalcString(val)) + SEP_FLAG;
  }
  if (info_.rfind(SEP_FLAG) == info_.size() - 1) {
    info_ = info_.substr(0, info_.size() - 1);
  }
  info_ += ARRAY_END_FLAG;
  info_ += SEP_FLAG;
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<void *> &v) {
  info_ += ARRAY_BEGIN_FLAG;
  for (auto p : v) {
    std::stringstream ss;
    ss << p;
    info_ += ss.str() + SEP_FLAG;
  }
  if (info_.rfind(SEP_FLAG) == info_.size() - 1) {
    info_ = info_.substr(0, info_.size() - 1);
  }
  info_ += ARRAY_END_FLAG;
  info_ += SEP_FLAG;
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<PyObject *> &v) {
  info_ += ARRAY_BEGIN_FLAG;
  for (auto p : v) {
    info_ += std::to_string(p != nullptr ? 1 : 0) + SEP_FLAG;  
    if (p != nullptr) {
        info_ += std::to_string(CalcString(std::string(py::str(p)))) + SEP_FLAG;
    }
  }
  if (info_.rfind(SEP_FLAG) == info_.size() - 1) {
    info_ = info_.substr(0, info_.size() - 1);
  }
  info_ += ARRAY_END_FLAG;
  info_ += SEP_FLAG;
  return *this;  
}

InfoPack &InfoPack::operator<<(const InfoPack &v) {
  info_ += v.ToString() + SEP_FLAG;
  return *this;
}

size_t InfoPack::CalcHash(std::string v) { return std::hash<std::string>()(v); }

class String2Id {
 public:
  String2Id() = default;
  ~String2Id() = default;
  size_t Insert(std::string key) {
    if (map_.find(key) == map_.end()) {
      map_[key] = map_.size();
      return map_.size() - 1;
    } else {
      return map_[key];
    }
  }

 protected:
  std::map<std::string, size_t> map_;
};

static String2Id g_IdMap;

size_t InfoPack::CalcId(std::string v) { return g_IdMap.Insert(v); }

static String2Id g_StrMap;

size_t InfoPack::CalcString(std::string v) { return g_StrMap.Insert(v); }

}  // namespace pijit
}  // namespace mindspore
