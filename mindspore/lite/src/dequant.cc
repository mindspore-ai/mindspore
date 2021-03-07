/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <string>
#include <memory>
#include "src/dequant.h"
#include "src/huffman_decode.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::lite {
float *DequantUtil::DequantWeight(lite::Tensor *input_tensor, bool channel_first) {
  MS_ASSERT(input_tensor != nullptr);
  if (input_tensor->data_type() != kNumberTypeInt8 && input_tensor->data_type() != kNumberTypeInt16) {
    MS_LOG(ERROR) << "Conv weight input type error." << input_tensor->data_type();
    return nullptr;
  }
  if (input_tensor->quant_params().empty()) {
    MS_LOG(ERROR) << "No quant param.";
    return nullptr;
  }
  if (input_tensor->data_type() == kNumberTypeInt16) {
    return DequantData<int16_t>(input_tensor, channel_first);
  } else {
    return DequantData<int8_t>(input_tensor, channel_first);
  }
}

int DequantUtil::UnPackToInt(const schema::Tensor *input_tensor, void *unpack_int_data) {
  MS_ASSERT(input_tensor != nullptr);
  MS_ASSERT(unpack_int_data != nullptr);
  auto quant_params = input_tensor->quantParams();
  if (quant_params == nullptr) {
    MS_LOG(ERROR) << "low bits quantparams is empty.";
    return RET_ERROR;
  }
  auto enable_huffman_code = input_tensor->enableHuffmanCode();
  if (enable_huffman_code) {
    std::string encode_str(input_tensor->data()->begin(), input_tensor->data()->end());
    auto huffman_decode = std::make_unique<lite::HuffmanDecode>();
    auto ret = huffman_decode->DoHuffmanDecode(encode_str, unpack_int_data);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "DoHuffmanDecode failed.";
      return ret;
    }
    return RET_OK;
  }
  int origin_bit = quant_params->Get(0)->numBits();
  if (origin_bit < 8 && origin_bit > 0) {
    UnPackUtil<int8_t, uint8_t>(input_tensor, origin_bit, unpack_int_data);
  } else if (origin_bit < 16 && origin_bit > 8) {
    UnPackUtil<int16_t, uint16_t>(input_tensor, origin_bit, unpack_int_data);
  }
  return RET_OK;
}

std::map<Tensor *, std::pair<TypeId, void *>> DequantUtil::DequantTensor(OpParameter *op_param,
                                                                         const std::vector<Tensor *> &in_tensors,
                                                                         TypeId data_type, bool need_restore) {
  std::map<Tensor *, std::pair<TypeId, void *>> tensor_origin_data;
  if (data_type == TypeId::kNumberTypeFloat32 || data_type == TypeId::kNumberTypeFloat16) {
    auto input_i = 0;
    for (auto weight_tensor : in_tensors) {
      MS_ASSERT(weight_tensor != nullptr);
      input_i++;
      auto channel_first = true;
      if (op_param->type_ == schema::PrimitiveType_MatMul && weight_tensor->shape().size() == 2) {
        auto param = reinterpret_cast<MatMulParameter *>(op_param);
        if (input_i == 1) {
          channel_first = !param->a_transpose_;
        } else if (input_i == 2) {
          channel_first = param->b_transpose_;
        } else {
          MS_LOG(WARNING) << "unexpected input_i";
        }
      }

      auto *restore_data = weight_tensor->data_c();
      auto restore_type = weight_tensor->data_type();
      bool dequant_flag = !weight_tensor->quant_params().empty() && weight_tensor->quant_params().front().inited &&
                          restore_data != nullptr &&
                          (restore_type == kNumberTypeInt8 || restore_type == kNumberTypeInt16);
      if (dequant_flag) {
        auto *dequant_weight = DequantUtil::DequantWeight(weight_tensor, channel_first);
        if (dequant_weight == nullptr) {
          MS_LOG(ERROR) << "dequant data is nullptr.";
          return tensor_origin_data;
        }
        if (need_restore) {
          tensor_origin_data[weight_tensor] = {restore_type, restore_data};
        } else {
          weight_tensor->FreeData();
        }
        weight_tensor->set_data(dequant_weight);
        weight_tensor->set_data_type(kNumberTypeFloat32);
      }
    }
  }
  return tensor_origin_data;
}

void DequantUtil::RestoreTensorData(const std::map<Tensor *, std::pair<TypeId, void *>> &tensor_origin_data_map) {
  for (auto &kv : tensor_origin_data_map) {
    auto *tensor = kv.first;
    auto type_id = kv.second.first;
    auto data = kv.second.second;
    tensor->FreeData();
    tensor->set_data_type(type_id);
    tensor->set_data(data);
  }
}

}  // namespace mindspore::lite
