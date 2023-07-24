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

#include "src/extendrt/delegate/tensorrt/tensor_info.h"
#include <algorithm>
#include <numeric>
#include <memory>
#include <functional>
#include "include/api/kernel.h"
#include "src/common/utils.h"
#include "src/extendrt/utils/tensor_default_impl.h"
#include "src/tensor.h"

namespace mindspore::lite {
TensorInfo::TensorInfo(const std::string &name, mindspore::DataType type, const std::vector<int64_t> &shape,
                       mindspore::Format format, const void *data, size_t data_len,
                       const mindspore::tensor::TensorPtr &tensor_val)
    : tensor_val_(tensor_val) {
  impl_ = LiteTensorImpl::CreateTensorImpl(name, type, shape, nullptr, 0);
  if ((impl_ == nullptr) || (impl_->lite_tensor() == nullptr)) {
    MS_LOG(ERROR) << "Create tensor failed.";
  } else {
    impl_->SetFormat(format);
    if (impl_->DataSize() != data_len) {
      MS_LOG_INFO << "Tensor expect data size " << impl_->DataSize() << " != data len " << data_len
                  << ", shape: " << impl_->Shape() << ", dtype: " << impl_->DataType();
    }
    impl_->SetData(const_cast<void *>(data), false);
    if (data != nullptr) {
      impl_->SetCategory(shape.empty() ? CONST_SCALAR : CONST_TENSOR);
    }
  }
}

std::string TensorInfo::Name() const {
  if (impl_ == nullptr) {
    return "";
  }
  return impl_->Name();
}

mindspore::DataType TensorInfo::DataType() const {
  if (impl_ == nullptr) {
    return mindspore::DataType::kTypeUnknown;
  }
  return impl_->DataType();
}

mindspore::Format TensorInfo::format() const {
  if (impl_ == nullptr) {
    return DEFAULT_FORMAT;
  }
  return impl_->Format();
}

const std::vector<int64_t> &TensorInfo::Shape() const {
  static const std::vector<int64_t> empty_shape;
  if (impl_ == nullptr) {
    return empty_shape;
  }
  return impl_->Shape();
}

const void *TensorInfo::Data() const {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return impl_->MutableData();
}

void *TensorInfo::MutableData() {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return impl_->MutableData();
}

size_t TensorInfo::DataSize() const {
  if (impl_ == nullptr) {
    return 0;
  }
  return ElementNum() * DataTypeSize(static_cast<enum TypeId>(impl_->DataType()));
}

lite::Tensor *TensorInfo::LiteTensor() const {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return impl_->lite_tensor();
}

bool TensorInfo::IsConst() const {
  if (impl_ == nullptr) {
    return 0;
  }
  return impl_->IsConst();
}

size_t TensorInfo::item_size() const {
  if (impl_ == nullptr) {
    return 0;
  }
  return DataTypeSize(static_cast<enum TypeId>(impl_->DataType()));
}

void TensorInfo::SetShape(const std::vector<int64_t> &shape) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->SetShape(shape);
}

void TensorInfo::SetDataType(const mindspore::DataType data_type) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->SetDataType(data_type);
}

void TensorInfo::SetData(const void *data, size_t data_len) {
  if (impl_ == nullptr) {
    return;
  }
  if (data != nullptr && data_len != 0) {
    if (impl_->DataSize() != data_len) {
      MS_LOG_WARNING << "Tensor expect data size " << impl_->DataSize() << " != data len " << data_len
                     << ", shape: " << impl_->Shape() << ", dtype: " << impl_->DataType();
    }
    impl_->SetData(const_cast<void *>(data), false);
  }
}

int64_t TensorInfo::ElementNum() const {
  if (impl_ == nullptr) {
    return 0;
  }
  return impl_->ElementNum();
}

TensorInfo &TensorInfo::operator=(const TensorInfo &other) {
  impl_ = other.impl_;
  return *this;
}

bool TensorInfo::operator==(const TensorInfo &other) const { return impl_ == other.impl_; }

bool TensorInfo::operator!=(const TensorInfo &other) const { return impl_ != other.impl_; }

bool TensorInfo::operator<(const TensorInfo &other) const { return impl_ < other.impl_; }
}  // namespace mindspore::lite
