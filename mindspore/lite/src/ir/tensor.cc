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
#include <string>
#include <utility>
#include "src/ir/tensor.h"
#include "securec/include/securec.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
namespace tensor {
#define kMaxMallocSize 1024 * 1024 * 100
Tensor::Tensor(const TypeId data_type, const std::vector<int> &shape, const schema::Format &format,
               schema::NodeType tensorType)
    : MetaTensor(data_type, shape), format_(format), tensorType(tensorType) {}

Tensor::Tensor(const Tensor &tensor) : MetaTensor(tensor) {
  auto ret = CopyTensor(tensor, true);
  if (0 != ret) {
    MS_LOG(EXCEPTION) << "CopyTensorData error";
  }
}

int Tensor::CopyTensorData(const Tensor &srcTensor) {
  if (srcTensor.data_ == nullptr) {
    MS_LOG(ERROR) << "data of srcTensor is nullptr";
    return mindspore::lite::RET_PARAM_INVALID;
  }
  size_t data_size = this->Size();
  MS_ASSERT(data_size == srcTensor.Size());
  if (this->data_ == nullptr) {
    if (data_size > kMaxMallocSize) {
      MS_LOG(ERROR) << "Malloc size is too big while coping data, " << data_size << " bytes";
      return mindspore::lite::RET_ERROR;
    }
    this->data_ = malloc(data_size);
  }
  memcpy(this->data_, srcTensor.data_, data_size);
  return 0;
}

int Tensor::CopyTensor(const Tensor &srcTensor, bool copyData) {
  this->data_type_ = srcTensor.data_type_;
  this->shape_ = srcTensor.shape_;
  this->tensorType = srcTensor.tensorType;
  if (copyData) {
    auto ret = CopyTensorData(srcTensor);
    if (0 != ret) {
      MS_LOG(ERROR) << "CopyTensorData error";
      return mindspore::lite::RET_ERROR;
    }
  }
  return 0;
}

Tensor::~Tensor() {
  if (nullptr != this->data_) {
    if (this->allocator_ != nullptr) {
      this->allocator_->Free(this->data_);
    } else {
      free(this->data_);
    }
  }
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  if (&tensor == this) {
    return *this;
  }
  auto ret = CopyTensor(tensor, true);
  if (0 != ret) {
    MS_LOG(ERROR) << "CopyTensorData error";
    MS_ASSERT(false);
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

int32_t Tensor::Batch() const {
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return -1;
  }
  switch (this->format_) {
    case schema::Format_NHWC:
    case schema::Format_NHWC4:
    case schema::Format_NCHW:
    case schema::Format_NC4HW4:
    case schema::Format_KCHW:
    case schema::Format_KHWC:
    case schema::Format_NC:
    case schema::Format_NC4:
      return this->shape_[0];
    case schema::Format_HWCK:
    case schema::Format_CHWK:
      return this->shape_[3];
    case schema::Format_HWKC:
      return this->shape_[2];
    case schema::Format_CKHW:
      return this->shape_[1];
    default:
      MS_LOG(ERROR) << "Unsupported format: " << schema::EnumNameFormat(this->format_);
      return -1;
  }
}

int32_t Tensor::Channel() const {
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return -1;
  }
  switch (this->format_) {
    case schema::Format_NCHW:
    case schema::Format_KCHW:
    case schema::Format_NC:
    case schema::Format_NC4:
      return this->shape_[1];
    case schema::Format_HWCK:
      return this->shape_[2];
    case schema::Format_HWKC:
    case schema::Format_NHWC:
    case schema::Format_NHWC4:
    case schema::Format_NC4HW4:
    case schema::Format_KHWC:
      return this->shape_[3];
    case schema::Format_CKHW:
    case schema::Format_CHWK:
      return this->shape_[0];
    default:
      return -1;
  }
}

int32_t Tensor::Height() const {
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return -1;
  }
  switch (this->format_) {
    case schema::Format_NCHW:
    case schema::Format_KCHW:
    case schema::Format_CKHW:
      return this->shape_[2];
    case schema::Format_NHWC:
    case schema::Format_NHWC4:
    case schema::Format_NC4HW4:
    case schema::Format_KHWC:
    case schema::Format_CHWK:
      return this->shape_[1];
    case schema::Format_HWCK:
    case schema::Format_HWKC:
    case schema::Format_HW:
    case schema::Format_HW4:
      return this->shape_[0];
    default:
      MS_LOG(ERROR) << "Unsupported format: " << schema::EnumNameFormat(this->format_);
      return -1;
  }
}

int32_t Tensor::Width() const {
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return -1;
  }
  switch (this->format_) {
    case schema::Format_NCHW:
    case schema::Format_KCHW:
    case schema::Format_CKHW:
      return this->shape_[3];
    case schema::Format_KHWC:
    case schema::Format_NHWC:
    case schema::Format_NHWC4:
    case schema::Format_NC4HW4:
    case schema::Format_CHWK:
      return this->shape_[2];
    case schema::Format_HWCK:
    case schema::Format_HWKC:
    case schema::Format_HW:
    case schema::Format_HW4:
      return this->shape_[1];
    default:
      return -1;
  }
}

int32_t Tensor::ElementsC4Num() const {
  int32_t result = 0;
  if (this->shape_.size() == 4) {
    result = Batch() * Height() * Width() * ((Channel() + 3) / 4 * 4);
  } else if (this->shape_.size() == 2) {
    result = this->shape_[0] * ((this->shape_[1] + 3) / 4 * 4);
  }
  return result;
}

std::string Tensor::ToString() const {
  std::ostringstream oss;
  oss << "Format: " << schema::EnumNameFormat(this->format_);
  oss << " DataType: " << this->data_type_;
  oss << " NodeType: " << schema::EnumNameNodeType(this->tensorType);
  oss << " Shape:";
  for (auto &dim : this->shape()) {
    oss << " " << dim;
  }
  oss << std::endl << "Data:";
  switch (this->data_type_) {
    case kNumberTypeFloat32: {
      auto data = static_cast<float *>(this->data_);
      if (data == nullptr) {
        return "Data of tensor is nullptr";
      } else {
        for (size_t i = 0; i < 40 && i < this->ElementsNum(); i++) {
          oss << " " << data[i];
        }
      }
    } break;
    case kNumberTypeInt32: {
      auto data = static_cast<int32_t *>(this->data_);
      if (data == nullptr) {
        return "Data of tensor is nullptr";
      } else {
        for (size_t i = 0; i < 40 && i < this->ElementsNum(); i++) {
          oss << " " << data[i];
        }
      }
    } break;
    default:
      oss << "Unsupported data type to print";
      break;
  }
  return oss.str();
}

void Tensor::AddQuantParam(const tensor::QuantArg &quant_arg) { this->quant_params_.push_back(quant_arg); }

std::vector<tensor::QuantArg> Tensor::GetQuantParams() const { return this->quant_params_; }

LiteTensor::LiteTensor() { this->tensor_impl_ = new tensor::Tensor(); }

LiteTensor::LiteTensor(TypeId data_type, const std::vector<int> &shape) {
  this->tensor_impl_ = new tensor::Tensor(data_type, shape);
}

LiteTensor::LiteTensor(tensor::Tensor *tensor_ptr) { this->tensor_impl_ = tensor_ptr; }

TypeId LiteTensor::data_type() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->data_type();
}

TypeId LiteTensor::set_data_type(TypeId data_type) {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->set_data_type(data_type);
}

std::vector<int> LiteTensor::shape() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->shape();
}

size_t LiteTensor::set_shape(const std::vector<int> &shape) {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->set_shape(shape);
}

int LiteTensor::DimensionSize(size_t index) const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->DimensionSize(index);
}

int LiteTensor::ElementsNum() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->ElementsNum();
}

std::size_t LiteTensor::hash() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->hash();
}

tensor::Tensor *LiteTensor::tensor() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_;
}

size_t LiteTensor::Size() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->Size();
}

void *LiteTensor::MutableData() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  auto data = this->tensor_impl_->Data();
  if (nullptr == data) {
    auto ret = tensor_impl_->MallocData();
    if (0 != ret) {
      return nullptr;
    }
  }
  return this->tensor_impl_->Data();
}
LiteTensor::~LiteTensor() { delete this->tensor_impl_; }

void LiteTensor::SetTensorImpl(tensor::Tensor *tensor) { this->tensor_impl_ = tensor; }
}  // namespace tensor
}  // namespace lite
namespace tensor {
MSTensor *MSTensor::CreateTensor(TypeId data_type, const std::vector<int> &shape) {
  return new mindspore::lite::tensor::LiteTensor(data_type, shape);
}
}  // namespace tensor
}  // namespace mindspore
