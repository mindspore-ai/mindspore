/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER__GENERAL_BITPACKING_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER__GENERAL_BITPACKING_H
#include <stdint.h>
#include <stack>
#include <queue>
#include <vector>
#include <cassert>

namespace mindspore::lite {
class BitPack {
 public:
  ~BitPack() = default;

  template <typename T1, typename T2>
  static void BitPacking(int bit_num, const std::vector<T1> &origin_data_vec, std::vector<T2> *packed_data_vec) {
    std::stack<bool> bit_data_vec;
    for (size_t i = 0; i < origin_data_vec.size(); i++) {
      T2 tmp = origin_data_vec[i] + static_cast<T2>(pow(2, bit_num - 1));
      DoBinary<T2>(bit_num, tmp, &bit_data_vec, packed_data_vec);
    }
    size_t remain_bit_data = bit_data_vec.size();
    if (sizeof(T1) * 8 > remain_bit_data && remain_bit_data > 0) {
      for (size_t i = 0; i < sizeof(T1) * 8 - remain_bit_data; i++) {
        bit_data_vec.push(0);
      }
      PackFromOriginToUint<T2>(&bit_data_vec, packed_data_vec);
    }
  }

 private:
  template <typename T2>
  static void PackFromOriginToUint(std::stack<bool> *ans, std::vector<T2> *packed_data_vec) {
    uint32_t result = 0;
    for (size_t i = 0; i < sizeof(T2) * 8; i++) {
      bool bit_tmp = ans->top();
      result = (result << 1) + static_cast<int>(bit_tmp);
      ans->pop();
    }
    packed_data_vec->push_back(result);
  }

  template <typename T2>
  static void DoBinary(int bin_num, T2 n, std::stack<bool> *ans, std::vector<T2> *packed_data_vec) {
    for (int bit_count = 0; bit_count < bin_num; bit_count++) {
      bool a = n % 2;
      n = n / 2;
      ans->push(a);
      if (ans->size() == sizeof(T2) * 8) {
        PackFromOriginToUint(ans, packed_data_vec);
      }
    }
  }
};
}  // namespace mindspore::lite
#endif
