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

KernelTensorValue::KernelTensorValue(size_t size, const TypePtr &t) : Value(t) {
  if (t) {
    obj_type_id_ = t->object_type();
  }
  mutable_data_ = std::shared_ptr<uint8_t[]>(new (std::nothrow) uint8_t[size]);
  size_ = size;
  use_mutable_storage_ = true;
}

KernelTensorValue::KernelTensorValue(const void *data, size_t size, const TypePtr &t) : Value(t) {
  if (t) {
    obj_type_id_ = t->object_type();
  }
  mutable_data_ = data;
  size_ = size;
  use_mutable_storage_ = true;
}

KernelTensorValue::KernelTensorValue(const tensor::TensorDataPtr &tensor_data, const TypePtr &t) : Value(t) {
  const_data_ = tensor_data;
  obj_type_id_ = kObjectTypeTensorType;
}

KernelTensorValue::KernelTensorValue(std::vector<uint8_t> &&array_data, const TypePtr &t) : Value(t) {
  const_data_ = std::move(array_data);
  obj_type_id_ = kObjectTypeTuple;
}

KernelTensorValue::KernelTensorValue(const StringImmPtr &string, const TypePtr &t) : Value(t) {
  const_data_ = string;
  obj_type_id_ = kObjectTypeString;
}

bool KernelTensorValue::operator==(const Value &other) const {
  if (other.isa<KernelTensorValue>()) {
    return *this == static_cast<const KernelTensorValue &>(other);
  } else {
    return false;
  }
}

bool KernelTensorValue::operator==(const KernelTensorValue &other) const {
  if (use_mutable_storage_) {
    return mutable_data_ == other.mutable_data_;
  }
  return const_data_ == other.const_data_;
}

void *KernelTensorValue::GetMutableDataPtr() {
  if (use_mutable_storage_ && std::holds_alternative<std::shared_ptr<uint8_t[]>>(mutable_data_)) {
    return std::get<std::shared_ptr<uint8_t[]>>(mutable_data_).get();
  }
  MS_LOG(EXCEPTION) << "Can not get mutable data pointer for read-only KernelTensorValue.";
}

const void *KernelTensorValue::GetDataPtr() const {
  if (use_mutable_storage_) {
    if (std::holds_alternative<std::shared_ptr<uint8_t[]>>(mutable_data_)) {
      return std::get<std::shared_ptr<uint8_t[]>>(mutable_data_).get();
    }
    return std::get<const void *>(mutable_data_);
  }

  switch (obj_type_id_) {
    case kObjectTypeNumber:
    case kObjectTypeTuple: {
      const std::vector<uint8_t> &data = std::get<std::vector<uint8_t>>(const_data_);
      if (data.empty()) {
        return nullptr;
      }
      return data.data();
    }

    case kObjectTypeTensorType: {
      const tensor::TensorDataPtr &tensor_data = std::get<tensor::TensorDataPtr>(const_data_);
      MS_EXCEPTION_IF_NULL(tensor_data);
      return tensor_data->data();
    }

    case kObjectTypeString: {
      const StringImmPtr &string_imm = std::get<StringImmPtr>(const_data_);
      MS_EXCEPTION_IF_NULL(string_imm);
      return string_imm->value().data();
    }

    default:
      MS_LOG(EXCEPTION) << "Can not get data pointer for type: " << TypeIdLabel(obj_type_id_);
  }
}

size_t KernelTensorValue::GetDataSize() const {
  if (use_mutable_storage_) {
    return size_;
  }

  switch (obj_type_id_) {
    case kObjectTypeNumber:
    case kObjectTypeTuple: {
      const std::vector<uint8_t> &data = std::get<std::vector<uint8_t>>(const_data_);
      if (data.empty()) {
        return 0;
      }
      return data.size();
    }

    case kObjectTypeTensorType: {
      const tensor::TensorDataPtr &tensor_data = std::get<tensor::TensorDataPtr>(const_data_);
      MS_EXCEPTION_IF_NULL(tensor_data);
      return tensor_data->nbytes();
    }

    case kObjectTypeString: {
      const StringImmPtr &string_imm = std::get<StringImmPtr>(const_data_);
      MS_EXCEPTION_IF_NULL(string_imm);
      return string_imm->value().size();
    }

    default:
      MS_LOG(EXCEPTION) << "Can not get data size for type: " << TypeIdLabel(obj_type_id_);
  }
}

void KernelTensorValue::SetDataPtr(const void *data_ptr) {
  MS_EXCEPTION_IF_NULL(data_ptr);
  if (!use_mutable_storage_) {
    MS_LOG(EXCEPTION) << "Can not set data for const KernelTensorValue.";
  }
  if (std::holds_alternative<const void *>(mutable_data_)) {
    mutable_data_ = data_ptr;
    return;
  }

  MS_LOG(EXCEPTION) << "Can not set data pointer for KernelTensorValue which uses shared pointer to storage data.";
}

void KernelTensorValue::Resize(size_t size) {
  if (!use_mutable_storage_) {
    MS_LOG(EXCEPTION) << "Can not resize const KernelTensorValue.";
  }

  if (std::holds_alternative<std::shared_ptr<uint8_t[]>>(mutable_data_)) {
    if (size_ < size) {
      mutable_data_ = std::shared_ptr<uint8_t[]>(new (std::nothrow) uint8_t[size]);
    }
  }
  size_ = size;
}
}  // namespace mindspore
