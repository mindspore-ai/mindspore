/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_HASHING_H_
#define MINDSPORE_CORE_UTILS_HASHING_H_

#include <initializer_list>
#include <memory>

namespace mindspore {
inline std::size_t hash_combine(std::size_t hash_sum, std::size_t hash_val) {
  // Reference from http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0814r0.pdf
  return ((hash_sum << 6) + (hash_sum >> 2) + 0x9e3779b9 + hash_val) ^ hash_sum;
}

inline std::size_t hash_combine(const std::initializer_list<std::size_t> &hash_vals) {
  std::size_t hash_sum = 0;
  for (auto hash_val : hash_vals) {
    hash_sum = hash_combine(hash_sum, hash_val);
  }
  return hash_sum;
}

template <typename T>
struct PointerHash {
  constexpr std::size_t operator()(const T *ptr) const noexcept { return reinterpret_cast<std::size_t>(ptr); }
};

template <typename T>
struct PointerHash<std::shared_ptr<T>> {
  constexpr std::size_t operator()(const std::shared_ptr<T> &ptr) const noexcept {
    return reinterpret_cast<std::size_t>(ptr.get());
  }
};

// Generate hash code for a string literal at compile time.
// We using Java string hash algorithm here.
constexpr uint32_t ConstStringHash(const char *str) {
  uint32_t hash = 0;
  while (*str) {
    hash = hash * 31 + static_cast<uint32_t>(*str++);
  }
  return hash;
}

}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_HASHING_H_
