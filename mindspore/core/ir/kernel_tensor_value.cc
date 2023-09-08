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

#include "ir/kernel_tensor_value.h"

namespace mindspore {

KernelTensorValue::KernelTensorValue(const tensor::TensorDataPtr &tensor_data, const TypePtr &t) : Value(t) {
  data_ = tensor_data;
  obj_type_id_ = kObjectTypeTensorType;
}

KernelTensorValue::KernelTensorValue(std::vector<uint8_t> &&array_data, const TypePtr &t) : Value(t) {
  data_ = std::move(array_data);
  obj_type_id_ = kObjectTypeTuple;
}

KernelTensorValue::KernelTensorValue(const StringImmPtr &string, const TypePtr &t) : Value(t) {
  data_ = string;
  obj_type_id_ = kObjectTypeString;
}

bool KernelTensorValue::operator==(const Value &other) const {
  if (other.isa<KernelTensorValue>()) {
    return *this == static_cast<const KernelTensorValue &>(other);
  } else {
    return false;
  }
}

bool KernelTensorValue::operator==(const KernelTensorValue &other) const { return data_ == other.data_; }

const void *KernelTensorValue::GetDataPtr() const {
  switch (obj_type_id_) {
    case kObjectTypeNumber:
    case kObjectTypeTuple: {
      const std::vector<uint8_t> &data = std::get<std::vector<uint8_t>>(data_);
      if (data.empty()) {
        return nullptr;
      }
      return data.data();
    }

    case kObjectTypeTensorType: {
      const tensor::TensorDataPtr &tensor_data = std::get<tensor::TensorDataPtr>(data_);
      MS_EXCEPTION_IF_NULL(tensor_data);
      return tensor_data->data();
    }

    case kObjectTypeString: {
      const StringImmPtr &string_imm = std::get<StringImmPtr>(data_);
      MS_EXCEPTION_IF_NULL(string_imm);
      return string_imm->value().data();
    }

    default:
      MS_LOG(EXCEPTION) << "Can not get data pointer for type: " << TypeIdLabel(obj_type_id_);
  }
}

size_t KernelTensorValue::GetDataSize() const {
  switch (obj_type_id_) {
    case kObjectTypeNumber:
    case kObjectTypeTuple: {
      const std::vector<uint8_t> &data = std::get<std::vector<uint8_t>>(data_);
      if (data.empty()) {
        return 0;
      }
      return data.size();
    }

    case kObjectTypeTensorType: {
      const tensor::TensorDataPtr &tensor_data = std::get<tensor::TensorDataPtr>(data_);
      MS_EXCEPTION_IF_NULL(tensor_data);
      return tensor_data->nbytes();
    }

    case kObjectTypeString: {
      const StringImmPtr &string_imm = std::get<StringImmPtr>(data_);
      MS_EXCEPTION_IF_NULL(string_imm);
      return string_imm->value().size();
    }

    default:
      MS_LOG(EXCEPTION) << "Can not get data size for type: " << TypeIdLabel(obj_type_id_);
  }
}
}  // namespace mindspore
