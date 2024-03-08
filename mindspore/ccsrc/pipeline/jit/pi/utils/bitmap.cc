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
#include "pipeline/jit/pi/utils/bitmap.h"

namespace mindspore {
namespace pijit {

BitMap::Iter::Iter(const BitMap *map, bool begin) {
  data_ = map->data() + (begin ? 0 : map->count());
  end_ = map->data() + map->count();
  offset_ = (data_ - map->data()) << shf;
  NextOne();
}

BitMap::Iter &BitMap::Iter::operator++() {
  size_t tail = offset_ & mod;
  size_t mask = (*data_) >> (tail + 1);
  if (tail == mod || mask == 0) {
    offset_ = offset_ - tail + CountTrailingZeros(0);
    data_++;
    NextOne();
  } else {
    offset_ += CountTrailingZeros(mask) + 1;
  }
  return *this;
}

void BitMap::Iter::NextOne() {
  while (data_ != end_) {
    offset_ += CountTrailingZeros(*data_);
    if (*data_ != 0) {
      break;
    }
    ++data_;
  }
}

// logic and, operator &=()
void BitMap::And(const BitMap &o) {
  const size_t siz = std::min(count(), o.count());
  for (size_t i = 0; i < siz; ++i) {
    data()[i] &= o.data()[i];
  }
}

// logic or, operator |=()
void BitMap::Or(const BitMap &o) {
  const size_t siz = std::min(count(), o.count());
  for (size_t i = 0; i < siz; ++i) {
    data()[i] |= o.data()[i];
  }
}

bool BitMap::OrWithChange(const BitMap &o) {
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

void BitMap::Diff(const BitMap &o) {
  const size_t siz = std::min(count(), o.count());
  for (size_t i = 0; i < siz; ++i) {
    data()[i] &= ~o.data()[i];
  }
}

size_t BitMap::CountBits() const {
  const unsigned *begin = reinterpret_cast<const unsigned *>(data());
  const unsigned *end = reinterpret_cast<const unsigned *>(data() + count());
  return std::accumulate(begin, end, 0, [](size_t c, unsigned i) { return c + PopCount(i); });
}

}  // namespace pijit
}  // namespace mindspore
