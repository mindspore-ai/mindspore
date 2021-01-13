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
#include "src/dequant.h"

namespace mindspore::lite {
float *DequantUtil::DequantWeight(lite::Tensor *input_tensor) {
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
    return DequantData<int16_t>(input_tensor);
  } else {
    return DequantData<int8_t>(input_tensor);
  }
}

void DequantUtil::UnPackToInt(const schema::Tensor *input_tensor, void *unpack_int_data) {
  MS_ASSERT(input_tensor != nullptr);
  MS_ASSERT(unpack_int_data != nullptr);
  auto quant_params = input_tensor->quantParams();
  if (quant_params == nullptr) {
    MS_LOG(ERROR) << "low bits quantparams is empty.";
    return;
  }
  int origin_bit = quant_params->Get(0)->numBits();
  if (origin_bit < 8 && origin_bit > 0) {
    UnPackUtil<int8_t, uint8_t>(input_tensor, origin_bit, unpack_int_data);
  } else if (origin_bit < 16 && origin_bit > 8) {
    UnPackUtil<int16_t, uint16_t>(input_tensor, origin_bit, unpack_int_data);
  }
}

std::map<Tensor *, std::pair<TypeId, void *>> DequantUtil::DequantTensor(const std::vector<Tensor *> &in_tensors,
                                                                         TypeId data_type) {
  std::map<Tensor *, std::pair<TypeId, void *>> tensor_origin_data;
  if (data_type == TypeId::kNumberTypeFloat32 || data_type == TypeId::kNumberTypeFloat16) {
    for (auto weight_tensor : in_tensors) {
      MS_ASSERT(weight_tensor != nullptr);
      auto *restore_data = weight_tensor->data_c();
      auto restore_type = weight_tensor->data_type();
      bool dequant_flag = !weight_tensor->quant_params().empty() && weight_tensor->quant_params().front().inited &&
                          restore_data != nullptr;
      if (dequant_flag) {
        auto *dequant_weight = DequantUtil::DequantWeight(weight_tensor);
        if (dequant_weight == nullptr) {
          MS_LOG(ERROR) << "dequant data is nullptr.";
          return tensor_origin_data;
        }
        weight_tensor->set_data(dequant_weight);
        weight_tensor->set_data_type(kNumberTypeFloat32);
        tensor_origin_data[weight_tensor] = {restore_type, restore_data};
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
