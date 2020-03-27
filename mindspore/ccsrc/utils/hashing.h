/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_HASHING_H_
#define MINDSPORE_CCSRC_UTILS_HASHING_H_

#include <initializer_list>

namespace mindspore {
inline std::size_t hash_combine(std::size_t hash_sum, std::size_t hash_val) {
  // Reference from http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0814r0.pdf
  return ((hash_sum << 6) + (hash_sum >> 2) + 0x9e3779b9 + hash_val) ^ hash_sum;
}

inline std::size_t hash_combine(const std::initializer_list<std::size_t>& hash_vals) {
  std::size_t hash_sum = 0;
  for (auto hash_val : hash_vals) {
    hash_sum = hash_combine(hash_sum, hash_val);
  }
  return hash_sum;
}
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_HASHING_H_
