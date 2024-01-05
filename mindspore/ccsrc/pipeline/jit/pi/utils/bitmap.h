/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PI_UTILS_BITMAP_H
#define MINDSPORE_CCSRC_PIPELINE_JIT_PI_UTILS_BITMAP_H

#include <vector>
#include <numeric>
#include <algorithm>

namespace mindspore {
namespace jit {
namespace graph {

constexpr int popcount(unsigned x) {
#ifdef __GNUC__
  return __builtin_popcount(x);
#else
  int c = 0;
  while (x) {
    c += (x & 1);
    x >>= 1;
  }
  return c;
#endif
}

class BitMap {
 public:
  BitMap() : size_(0), bits_() {}
  explicit BitMap(size_t size) : size_(size), bits_(count(), 0) {}
  size_t size() const { return size_; }
  bool Get(size_t i) const { return data()[i >> shf] & (size_t(1) << (i & mod)); }
  void Set(size_t i) { data()[i >> shf] |= (size_t(1) << (i & mod)); }
  void Clear(size_t i) { data()[i >> shf] &= ~(size_t(1) << (i & mod)); }

  // logic and, operator &=()
  void And(const BitMap &o) {
    const size_t siz = std::min(count(), o.count());
    for (size_t i = 0; i < siz; ++i) {
      data()[i] &= o.data()[i];
    }
  }

  // logic or, operator |=()
  void Or(const BitMap &o) {
    const size_t siz = std::min(count(), o.count());
    for (size_t i = 0; i < siz; ++i) {
      data()[i] |= o.data()[i];
    }
  }

  bool OrWithChange(const BitMap &o) {
    const size_t siz = std::min(count(), o.count());
    bool change = false;
    for (size_t i = 0; i < siz; ++i) {
      auto a = data()[i];
      auto b = a | o.data()[i];
      data()[i] = b;
      change |= a != b;
    }
    return change;
  }

  void Diff(const BitMap &o) {
    const size_t siz = std::min(count(), o.count());
    for (size_t i = 0; i < siz; ++i) {
      data()[i] &= ~o.data()[i];
    }
  }

  size_t CountBits() const {
    const unsigned *begin = reinterpret_cast<const unsigned *>(data());
    const unsigned *end = reinterpret_cast<const unsigned *>(data() + count());
    return std::accumulate(begin, end, 0, [](size_t c, unsigned i) { return c + popcount(i); });
  }

 private:
  static constexpr const int shf = 6;
  static constexpr const int mod = (1 << shf) - 1;
  static_assert((1 << shf) == (sizeof(size_t) * 8));

  size_t count() const { return (size_ >> shf) + static_cast<bool>(size_ & mod); }
  size_t bytes() const { return count() * sizeof(size_t); }
  size_t *data() { return bits_.data(); }
  const size_t *data() const { return bits_.data(); }
  size_t size_;
  std::vector<size_t> bits_;
};

}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif
