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
void TensorCompressor::WriteBufferWithAlignByte(const std::vector<bool> &bool_vec, int8_t *data) {
  size_t shift = kBitNumPerByte;
  for (bool bit : bool_vec) {
    *data |= bit << (shift - 1);
    if (--shift == 0) {
      data++;
      shift = kBitNumPerByte;
    }
  }
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
    int shape_size;
    auto status = GetElementNumFromShape(tensor_input->dims, &shape_size);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Get ElementNum from shape failed.";
      return status;
    }
    std::vector<int16_t> origin_data(shape_size);
    status = memcpy_s(origin_data.data(), origin_data.size() * sizeof(int16_t), tensor_input->data.data(),
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

int TensorCompressor::SetNewCompressionTensor(const ParameterPtr &weight, const std::vector<bool> &bits, size_t bit_num,
                                              const tensor::TensorPtr &tensor_info,
                                              TensorCompressionType compression_type) {
  // Add New Tensor
  auto size_in_byte = static_cast<size_t>(ceil(bits.size() / kBitNumPerByte));
  std::shared_ptr<mindspore::tensor::Tensor> compression_tensor = nullptr;
  if (bit_num >= k1Bit && bit_num <= k8Bit) {
    compression_tensor = std::make_shared<mindspore::tensor::Tensor>(kNumberTypeInt8, tensor_info->shape(),
                                                                     size_in_byte, compression_type);
  } else if (bit_num > k8Bit && bit_num <= k16Bit) {
    compression_tensor = std::make_shared<mindspore::tensor::Tensor>(kNumberTypeInt16, tensor_info->shape(),
                                                                     size_in_byte, compression_type);
  } else {
    MS_LOG(ERROR) << "bit_num only support 1 ~ 16 bit.";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(compression_tensor);
  // update tensor data
  WriteBufferWithAlignByte(bits, static_cast<int8_t *>(compression_tensor->data().data()));
  weight->set_default_param(compression_tensor);
  weight->set_abstract(compression_tensor->ToAbstract());
  return RET_OK;
}

int TensorCompressor::DoBitPack(const ParameterPtr &weight, size_t bit_num) {
  auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
  CHECK_NULL_RETURN(tensor_info);
  auto elements_num = tensor_info->ElementsNum();
  std::shared_ptr<mindspore::tensor::Tensor> compression_tensor = nullptr;
  if (bit_num > 0 && bit_num < k8Bit) {
    auto quant_data = static_cast<int8_t *>(tensor_info->data().data());
    std::vector<int8_t> origin_data(quant_data, quant_data + elements_num);
    std::vector<uint8_t> pack_data{};
    BitPack::BitPacking<int8_t, uint8_t>(bit_num, origin_data, &pack_data);
    auto buffer_size = pack_data.size() * sizeof(int8_t);
    compression_tensor = std::make_shared<mindspore::tensor::Tensor>(kNumberTypeInt8, tensor_info->shape(), buffer_size,
                                                                     mindspore::kBitPacking);
    CHECK_NULL_RETURN(compression_tensor);
    auto ret = memcpy_s(compression_tensor->data_c(), buffer_size, pack_data.data(), buffer_size);
    if (ret != EOK) {
      MS_LOG(ERROR) << weight->name() << " memcpy failed.";
      return RET_ERROR;
    }
  } else if (bit_num > k8Bit && bit_num < k16Bit) {
    auto quant_data = static_cast<int16_t *>(tensor_info->data().data());
    std::vector<int16_t> origin_data(quant_data, quant_data + elements_num);
    std::vector<uint16_t> pack_data{};
    BitPack::BitPacking<int16_t, uint16_t>(bit_num, origin_data, &pack_data);
    auto buffer_size = pack_data.size() * sizeof(int16_t);
    compression_tensor = std::make_shared<mindspore::tensor::Tensor>(kNumberTypeInt16, tensor_info->shape(),
                                                                     buffer_size, mindspore::kBitPacking);
    CHECK_NULL_RETURN(compression_tensor);
    auto ret = memcpy_s(compression_tensor->data_c(), buffer_size, pack_data.data(), buffer_size);
    if (ret != EOK) {
      MS_LOG(ERROR) << weight->name() << " memcpy failed.";
      return RET_ERROR;
    }
  }
  // update tensor data
  weight->set_default_param(compression_tensor);
  weight->set_abstract(compression_tensor->ToAbstract());
  return RET_OK;
}
}  // namespace mindspore::lite::quant
