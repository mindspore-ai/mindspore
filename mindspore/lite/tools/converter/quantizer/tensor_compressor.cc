/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/tensor_compressor.h"
#include <memory>
#include <numeric>
#include <limits>
#include <string>
#include <vector>
#include <functional>
#include <set>
#include <map>
#include <algorithm>

namespace mindspore::lite::quant {
namespace {
constexpr size_t kBitNumPerByte = 8;
}
std::string TensorCompressor::BoolVectorToString(const std::vector<bool> &bool_vec) {
  size_t size_in_byte = static_cast<size_t>(ceil(bool_vec.size() / kBitNumPerByte));
  std::string str(size_in_byte, '\0');
  auto iter = str.begin();
  size_t shift = kBitNumPerByte;
  for (bool bit : bool_vec) {
    *iter |= bit << (shift - 1);
    if (--shift == 0) {
      iter++;
      shift = kBitNumPerByte;
    }
  }
  return str;
}

int TensorCompressor::DoBitPack(const size_t &bit_num, schema::TensorT *tensor_input) {
  if (bit_num > 0 && bit_num < k8Bit) {
    std::vector<int8_t> origin_data(tensor_input->data.size());
    auto status = memcpy_s(origin_data.data(), origin_data.size() * sizeof(int8_t), tensor_input->data.data(),
                           tensor_input->data.size() * sizeof(uint8_t));
    if (status != EOK) {
      MS_LOG(ERROR) << tensor_input->name << " memcpy failed. " << status;
      return RET_ERROR;
    }
    std::vector<uint8_t> pack_data{};
    BitPack::BitPacking<int8_t, uint8_t>(bit_num, origin_data, &pack_data);
    tensor_input->data.resize(pack_data.size() * sizeof(uint8_t));
    status = memcpy_s(tensor_input->data.data(), tensor_input->data.size() * sizeof(uint8_t), pack_data.data(),
                      pack_data.size() * sizeof(uint8_t));
    if (status != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed. " << status;
      return RET_ERROR;
    }
  } else if (bit_num > k8Bit && bit_num < k16Bit) {
    auto shape_size =
      std::accumulate(tensor_input->dims.begin(), tensor_input->dims.end(), size_t(1), std::multiplies<size_t>());
    std::vector<int16_t> origin_data(shape_size);
    auto status = memcpy_s(origin_data.data(), origin_data.size() * sizeof(int16_t), tensor_input->data.data(),
                           tensor_input->data.size() * sizeof(uint8_t));
    if (status != EOK) {
      MS_LOG(ERROR) << "memcpy failed. " << status;
      return RET_ERROR;
    }
    std::vector<uint16_t> pack_data{};
    BitPack::BitPacking<int16_t, uint16_t>(bit_num, origin_data, &pack_data);
    tensor_input->data.resize(pack_data.size() * sizeof(uint16_t));
    status = memcpy_s(tensor_input->data.data(), tensor_input->data.size() * sizeof(uint8_t), pack_data.data(),
                      pack_data.size() * sizeof(uint16_t));
    if (status != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed. " << status;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
