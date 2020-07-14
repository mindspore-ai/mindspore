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

#include <vector>
#include <utility>
#include "ir/lite/tensor.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace tensor {
#define kMaxMallocSize 1024 * 1024 * 100
Tensor::Tensor(const TypeId data_type, const std::vector<int> &shape) : MetaTensor(data_type, shape) {}

Tensor::Tensor(const TypePtr &type_ptr, const std::vector<int> &shape) : MetaTensor(type_ptr, shape) {}

Tensor::Tensor(const Tensor &tensor) : MetaTensor(tensor) {
  this->data_type_ = tensor.data_type_;
  this->shape_ = tensor.shape_;
  auto ret = CopyTensorData(tensor);
  if (0 != ret) {
    MS_LOG(EXCEPTION) << "CopyTensorData error";
  }
}

int Tensor::CopyTensorData(const Tensor &srcTensor) {
  if (srcTensor.data_ == nullptr) {
    MS_LOG(ERROR) << "data of srcTensor is nullptr";
    return -1;
  }
  size_t data_size = this->Size();
  MS_ASSERT(data_size == tensor.Size());
  if (this->data_ == nullptr) {
    if (data_size > kMaxMallocSize) {
      MS_LOG(ERROR) << "Malloc size is too big while coping data, " << data_size << " bytes";
      return -1;
    }
    this->data_ = malloc(data_size);
  }
  memcpy_s(this->data_, data_size, tensor.data_, tensor.Size());
  return 0;
}

Tensor::~Tensor() {
  if (nullptr != this->data_) {
    free(this->data_);
  }
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  if (&tensor == this) {
    return *this;
  }
  this->shape_ = tensor.shape_;
  this->data_type_ = tensor.data_type_;
  auto ret = CopyTensorData(tensor);
  if (0 != ret) {
    MS_LOG(EXCEPTION) << "CopyTensorData error";
  }
  return *this;
}

bool Tensor::operator==(const Tensor &tensor) {
  return data_ == tensor.data_ && shape_ == tensor.shape_ && data_type_ == tensor.data_type_;
}

bool Tensor::operator==(const Value &other) const {
  if (other.isa<Tensor>()) {
    auto other_ = static_cast<const Tensor &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
}  // namespace tensor

namespace inference {
MSTensor *MSTensor::CreateTensor(TypeId data_type, const std::vector<int> &shape) {
  return new Tensor(data_type, shape);
}

Tensor::Tensor() { this->tensor_impl_ = std::make_shared<tensor::Tensor>(); }

Tensor::Tensor(TypeId data_type, const std::vector<int> &shape) {
  this->tensor_impl_ = std::make_shared<tensor::Tensor>(data_type, shape);
}

Tensor::Tensor(std::shared_ptr<tensor::Tensor> tensor_ptr) { this->tensor_impl_ = std::move(tensor_ptr); }

TypeId Tensor::data_type() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->data_type();
}

TypeId Tensor::set_data_type(TypeId data_type) {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->set_data_type(data_type);
}

std::vector<int> Tensor::shape() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->shape();
}

size_t Tensor::set_shape(const std::vector<int> &shape) {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->set_shape(shape);
}

int Tensor::DimensionSize(size_t index) const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->DimensionSize(index);
}

int Tensor::ElementsNum() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->ElementsNum();
}

std::size_t Tensor::hash() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->hash();
}

std::shared_ptr<tensor::Tensor> Tensor::tensor() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_;
}

size_t Tensor::Size() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->Size();
}

void *Tensor::MutableData() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->data();
}
}  // namespace inference
}  // namespace mindspore
