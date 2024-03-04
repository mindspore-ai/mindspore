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
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include "pipeline/jit/pi/utils/utils.h"

namespace mindspore {
namespace pijit {

static constexpr char kSepFlag = '\\';
static constexpr char kBeginFlag = '{';
static constexpr char kEndFlag = '}';
static constexpr char kArrayBeginFlag = '[';
static constexpr char kArrayEndFlag = ']';
static constexpr size_t kInitLimit = 1024;

template <typename T>
size_t StoreScalar(uint8_t *buf, size_t ptr, T val) {
  uint8_t *pVal = reinterpret_cast<uint8_t *>(&val);
  constexpr int kScalarSize = sizeof(T);
  for (int idx = 0; idx < kScalarSize; ++idx) {
    buf[ptr++] = pVal[idx];
  }
  return ptr;
}

template <typename T>
size_t AppendScalar(uint8_t *buf, size_t ptr, T v) {
  ptr = StoreScalar(buf, ptr, v);
  ptr = StoreScalar(buf, ptr, kSepFlag);
  return ptr;
}

template <typename T>
size_t StoreVector(uint8_t *buf, size_t ptr, const std::vector<T> &val) {
  T *pVal = const_cast<T *>(val.data());
  size_t szVal = val.size() * sizeof(T);
  ptr = StoreScalar(buf, ptr, szVal);
  memcpy(buf + ptr, pVal, szVal);
  return ptr + szVal;
}

template <typename T>
size_t AppendVector(uint8_t *buf, size_t ptr, const std::vector<T> &v) {
  ptr = StoreScalar(buf, ptr, kArrayBeginFlag);
  ptr = StoreVector(buf, ptr, v);
  ptr = StoreScalar(buf, ptr, kArrayEndFlag);
  ptr = StoreScalar(buf, ptr, kSepFlag);
  return ptr;
}

InfoPack::InfoPack() : id_(kInvalidId), buf_(std::make_unique<uint8_t[]>(kInitLimit)), ptr_(0), limit_(kInitLimit) {}

InfoPack::InfoPack(const InfoPack &dup)
    : id_(dup.id_), buf_(std::make_unique<uint8_t[]>(dup.ptr_)), ptr_(dup.ptr_), limit_(dup.ptr_) {
  memcpy(buf_.get(), dup.buf_.get(), dup.ptr_);
}

InfoPack::~InfoPack() { buf_.reset(nullptr); }

size_t InfoPack::Id() const { return id_; }

uint8_t *InfoPack::Buf(size_t *sz) const {
  if (sz != nullptr) {
    *sz = ptr_;
    return buf_.get();
  }
  return nullptr;
}

void InfoPack::Update() { id_ = CalcBuffer(buf_.get(), ptr_); }

InfoPack &InfoPack::Begin() {
  AllocIfNeed(sizeof(kBeginFlag));
  *(buf_.get() + ptr_++) = (uint8_t)kBeginFlag;
  return *this;
}

InfoPack &InfoPack::End() {
  if (buf_.get()[ptr_ - 1] == kSepFlag) {
    buf_.get()[ptr_ - 1] = kEndFlag;
  } else {
    AllocIfNeed(sizeof(kEndFlag));
    *(buf_.get() + ptr_++) = (uint8_t)kEndFlag;
  }
  return *this;
}

InfoPack &InfoPack::operator<<(int8_t v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(uint8_t v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(int16_t v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(uint16_t v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(int32_t v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(uint32_t v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(int64_t v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(uint64_t v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(float v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(double v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(bool vv) {
  uint8_t v = vv ? 1 : 0;
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(void *v) {
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(PyObject *vv) {
  uint8_t v = vv != nullptr ? 1 : 0;
  if (vv != nullptr) {
    size_t w = CalcString(std::string(py::str(vv)));
    AllocIfNeed(sizeof(v) + sizeof(w) + sizeof(kSepFlag));
    ptr_ = StoreScalar(buf_.get(), ptr_, v);
    ptr_ = AppendScalar(buf_.get(), ptr_, w);
  } else {
    AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
    ptr_ = AppendScalar(buf_.get(), ptr_, v);
  }
  return *this;
}

InfoPack &InfoPack::operator<<(mindspore::BasePtr vv) {
  uint8_t v = vv != nullptr ? 1 : 0;
  if (vv != nullptr) {
    size_t w = CalcString(vv->ToString());
    AllocIfNeed(sizeof(v) + sizeof(w) + sizeof(kSepFlag));
    ptr_ = StoreScalar(buf_.get(), ptr_, v);
    ptr_ = AppendScalar(buf_.get(), ptr_, w);
  } else {
    AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
    ptr_ = AppendScalar(buf_.get(), ptr_, v);
  }
  return *this;
}

InfoPack &InfoPack::operator<<(const std::string &vv) {
  size_t v = CalcString(vv);
  AllocIfNeed(sizeof(v) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<int8_t> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<uint8_t> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<int16_t> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<uint16_t> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<int32_t> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<uint32_t> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<int64_t> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<uint64_t> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<float> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<double> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<bool> &vv) {
  std::vector<uint8_t> v;
  std::transform(vv.begin(), vv.end(), std::back_inserter(v), [](const auto &item) { return item ? 1 : 0; });
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<std::string> &vv) {
  std::vector<size_t> v;
  std::transform(vv.begin(), vv.end(), std::back_inserter(v), [this](const auto &item) { return CalcString(item); });
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<void *> &v) {
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const std::vector<PyObject *> &vv) {
  std::vector<size_t> v;
  std::transform(vv.begin(), vv.end(), std::back_inserter(v),
                 [this](const auto &item) { return CalcString(std::string(py::str(item))); });
  AllocIfNeed(sizeof(kArrayBeginFlag) + sizeof(kArrayEndFlag) + sizeof(kSepFlag) + sizeof(v[0]) * v.size() +
              sizeof(size_t));
  ptr_ = AppendVector(buf_.get(), ptr_, v);
  return *this;
}

InfoPack &InfoPack::operator<<(const InfoPack &v) {
  size_t id = v.Id();
  AllocIfNeed(sizeof(id) + sizeof(kSepFlag));
  ptr_ = AppendScalar(buf_.get(), ptr_, id);
  return *this;
}

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

static String2Id g_StrMap;

size_t InfoPack::CalcString(std::string v) { return g_StrMap.Insert(v); }

struct BufferHash {
  bool operator()(const std::vector<uint8_t> &lhs, const std::vector<uint8_t> &rhs) const {
    if (lhs.size() == rhs.size()) {
      return memcmp(lhs.data(), rhs.data(), lhs.size()) == 0;
    }
    return false;
  }
  size_t operator()(const std::vector<uint8_t> &k) const {
    size_t ret = 0;
    ret = std::accumulate(k.begin(), k.end(), ret, [](size_t key, uint8_t v) {
      static constexpr int kShiftKey = 3;
      return (key << kShiftKey) + v;
    });
    return ret;
  }
};

class Buffer2Id {
 public:
  Buffer2Id() = default;
  ~Buffer2Id() = default;
  size_t Insert(uint8_t *buf, size_t sz) {
    std::vector<uint8_t> vec(sz);
    memcpy(vec.data(), buf, sz);
    auto it = map_.find(vec);
    if (it == map_.end()) {
      size_t ret = map_.size();
      map_[vec] = ret;
      return ret;
    } else {
      return it->second;
    }
  }

 protected:
  std::unordered_map<std::vector<uint8_t>, size_t, BufferHash, BufferHash> map_;
};

static Buffer2Id g_BufMap;

size_t InfoPack::CalcBuffer(uint8_t *buf, size_t sz) { return g_BufMap.Insert(buf, sz); }

void InfoPack::AllocIfNeed(size_t need) {
  if (limit_ < need + ptr_) {
    do {
      limit_ += kInitLimit;
    } while (limit_ < need + ptr_);
    auto buf = std::make_unique<uint8_t[]>(limit_);
    memcpy(buf.get(), buf_.get(), ptr_ * sizeof(uint8_t));
    buf_.reset(buf.release());
  }
}
}  // namespace pijit
}  // namespace mindspore
