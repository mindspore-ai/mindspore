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
#ifndef MINDSPORE_PI_JIT_UTILS_BITMAP_H
#define MINDSPORE_PI_JIT_UTILS_BITMAP_H

#include <vector>
#include <numeric>
#include <limits>
#include <algorithm>

namespace mindspore {
namespace pijit {

constexpr int PopCount(unsigned x) {
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

constexpr int CountTrailingZeros(size_t x) {
  constexpr auto bits = std::numeric_limits<size_t>::digits;
  constexpr auto limit = std::numeric_limits<unsigned>::digits;
  if (x == 0) {
    return bits;
  }
  int c = 0;
  if ((x & std::numeric_limits<unsigned>::max()) == 0) {
    c += limit;
    x >>= limit;
  }
#ifdef __GNUC__
  return bits == limit ? __builtin_ctz(x) : c + __builtin_ctzll(x);
#else
  while ((x & 1) == 0) {
    c++;
    x >>= 1;
  }
  return c;
#endif
}

class BitMap {
 public:
  // traverse the index of bit that is set
  class Iter {
   public:
    Iter(const BitMap *map, bool begin);
    bool operator!=(const Iter &o) const { return this->data_ != o.data_; }
    size_t operator*() { return offset_; }
    Iter &operator++();

   private:
    // advance to next one bit
    void NextOne();

    const size_t *end_;
    const size_t *data_;
    size_t offset_;
  };

  BitMap() : size_(0), bits_() {}
  explicit BitMap(size_t size) : size_(size), bits_(count(), 0) {}
  size_t size() const { return size_; }
  bool Get(size_t i) const { return data()[i >> shf] & (size_t(1) << (i & mod)); }
  void Set(size_t i) { data()[i >> shf] |= (size_t(1) << (i & mod)); }
  void Clear(size_t i) { data()[i >> shf] &= ~(size_t(1) << (i & mod)); }

  // logic and, a &= b
  void And(const BitMap &o);

  // logic or, a |= b
  void Or(const BitMap &o);

  // logic or, return true if any bit changed
  bool OrWithChange(const BitMap &o);

  // logic and not, a &= ~b
  void Diff(const BitMap &o);

  // count one bits
  size_t CountBits() const;

 private:
  static constexpr const int shf = CountTrailingZeros(std::numeric_limits<size_t>::digits);
  static constexpr const int mod = (1 << shf) - 1;

  size_t count() const { return (size_ >> shf) + static_cast<bool>(size_ & mod); }
  size_t bytes() const { return count() * sizeof(size_t); }
  size_t *data() { return bits_.data(); }
  const size_t *data() const { return bits_.data(); }

  size_t size_;
  std::vector<size_t> bits_;
};

}  // namespace pijit
}  // namespace mindspore

#endif
