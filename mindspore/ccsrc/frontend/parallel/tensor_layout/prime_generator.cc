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
#include "frontend/parallel/tensor_layout/prime_generator.h"

namespace mindspore::parallel {
const int MAX_PRIME_RANGE = 1e5 + 1;  // 100,001
void get_prime_table(Shape *prime_arr, const size_t arr_size) {
  std::vector<bool> is_composite_num(arr_size, false);
  for (size_t i = 2; i <= arr_size; i++) {
    if (!is_composite_num[i]) {
      prime_arr->emplace_back(i);
    }
    for (size_t j = 0;; j++) {
      if (j >= prime_arr->size() || LongToSize(prime_arr->at(j)) * i > arr_size) {
        break;
      }
      is_composite_num[LongToSize(prime_arr->at(j)) * i] = true;
      if (i % LongToSize(prime_arr->at(j)) == 0) {
        break;
      }
    }
  }
  prime_arr->resize(prime_arr->size());
}

PrimeGenerator::PrimeGenerator() {
  this->prime_table_ = {3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,
                        61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137,
                        139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
                        229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293};
}

int64_t PrimeGenerator::GetCoprimeNum(const Shape &tensor_shape) {
  const int64_t unknown_val = -1;
  if (tensor_shape.empty()) {
    return this->prime_table_[0];
  }
  std::set<int64_t> input_flag;
  for (int64_t i : tensor_shape) {
    input_flag.insert(i);
  }
  const int64_t two = 2;
  for (int64_t prime_num : this->prime_table_) {
    if (prime_num == two) {
      // skip prime 2.
      continue;
    }
    if (input_flag.find(prime_num) != input_flag.end()) {
      continue;
    }
    bool is_coprime = std::all_of(tensor_shape.begin(), tensor_shape.end(),
                                  [prime_num](int64_t v) { return std::gcd(prime_num, v) == 1; });
    if (is_coprime) {
      return prime_num;
    }
  }
  MS_LOG(ERROR) << "Cannot find a coprime number for shape " << tensor_shape;
  return unknown_val;
}
}  // namespace mindspore::parallel
