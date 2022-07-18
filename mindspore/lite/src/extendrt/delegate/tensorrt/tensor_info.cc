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

namespace mindspore::lite {
class TensorInfoImpl {
 public:
  TensorInfoImpl() {}
  TensorInfoImpl(const std::string &name, mindspore::DataType type, const std::vector<int64_t> &shape,
                 mindspore::Format format, const void *data, size_t data_len)
      : name_(name), dType_(type), shape_(shape), format_(format), data_(data), data_len_(data_len) {
    is_const_ = (data_ != nullptr);
    if (data_ == nullptr || data_len_ == 0) {
      auto ele_num = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());
      auto type_size = DataTypeSize(static_cast<enum TypeId>(dType_));
      temp_data_.resize(ele_num * type_size);
      data_ = temp_data_.data();
      data_len_ = temp_data_.size();
    }
  }
  std::string name_;
  mindspore::DataType dType_ = mindspore::DataType::kTypeUnknown;
  std::vector<int64_t> shape_;
  mindspore::Format format_ = DEFAULT_FORMAT;
  const void *data_ = nullptr;
  size_t data_len_ = 0;
  bool is_const_ = false;
  std::vector<uint8_t> temp_data_;
};

TensorInfo::TensorInfo(const std::string &name, mindspore::DataType type, const std::vector<int64_t> &shape,
                       mindspore::Format format, const void *data, size_t data_len) {
  impl_ = std::make_shared<TensorInfoImpl>(name, type, shape, format, data, data_len);
}

std::string TensorInfo::Name() const {
  if (impl_ == nullptr) {
    return "";
  }
  return impl_->name_;
}

mindspore::DataType TensorInfo::DataType() const {
  if (impl_ == nullptr) {
    return mindspore::DataType::kTypeUnknown;
  }
  return impl_->dType_;
}

mindspore::Format TensorInfo::format() const {
  if (impl_ == nullptr) {
    return DEFAULT_FORMAT;
  }
  return impl_->format_;
}

const std::vector<int64_t> &TensorInfo::Shape() const {
  static const std::vector<int64_t> empty_shape;
  if (impl_ == nullptr) {
    return empty_shape;
  }
  return impl_->shape_;
}

const void *TensorInfo::Data() const {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return impl_->data_;
}

void *TensorInfo::MutableData() {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return const_cast<void *>(impl_->data_);
}

size_t TensorInfo::DataSize() const {
  if (impl_ == nullptr) {
    return 0;
  }
  return impl_->data_len_;
}

bool TensorInfo::IsConst() const {
  if (impl_ == nullptr) {
    return 0;
  }
  return impl_->is_const_ && impl_->data_ != nullptr;
}

size_t TensorInfo::item_size() const { return DataTypeSize(static_cast<enum TypeId>(DataType())); }

void TensorInfo::SetShape(const std::vector<int64_t> &shape) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->shape_ = shape;
}

void TensorInfo::SetData(const void *data, size_t data_len) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->data_ = data;
  impl_->data_len_ = data_len;
}

int64_t TensorInfo::ElementNum() const {
  if (impl_ == nullptr) {
    return 0;
  }
  if (impl_->shape_.empty()) {
    // element number of scalar is 1
    return 1;
  }
  return std::accumulate(impl_->shape_.begin(), impl_->shape_.end(), 1, std::multiplies<int64_t>());
}

TensorInfo &TensorInfo::operator=(const TensorInfo &other) {
  impl_ = other.impl_;
  return *this;
}

bool TensorInfo::operator==(const TensorInfo &other) const { return impl_ == other.impl_; }

bool TensorInfo::operator!=(const TensorInfo &other) const { return impl_ != other.impl_; }

bool TensorInfo::operator<(const TensorInfo &other) const { return impl_ < other.impl_; }
}  // namespace mindspore::lite
