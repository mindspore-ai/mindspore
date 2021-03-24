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
#include <algorithm>
#include <functional>
#include "src/tensor.h"
#include "securec/include/securec.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
#define kMaxMallocSize 1024 * 1024 * 100
Tensor::Tensor(const TypeId data_type, std::vector<int> shape, const schema::Format &format, Category category)
    : data_type_(data_type), shape_(std::move(shape)), format_(format), category_(category) {}

int Tensor::CopyTensorData(const Tensor &src_tensor, Tensor *dst_tensor) {
  if (dst_tensor == nullptr) {
    MS_LOG(ERROR) << "dst_tensor is nullptr";
    return RET_PARAM_INVALID;
  }
  if (src_tensor.data_ == nullptr) {
    MS_LOG(ERROR) << "data of src tensor is nullptr";
    return RET_PARAM_INVALID;
  }
  size_t data_size = dst_tensor->Size();
  if (data_size != src_tensor.Size()) {
    MS_LOG(ERROR) << "Size of dst tensor is not compatible with src tensor";
    return RET_ERROR;
  }
  if (dst_tensor->data_ == nullptr) {
    if (data_size > kMaxMallocSize) {
      MS_LOG(ERROR) << "Malloc size is too big while coping data, " << data_size << " bytes";
      return RET_ERROR;
    }
    dst_tensor->data_ = malloc(data_size);
    if (dst_tensor->data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc memory failed";
      return RET_ERROR;
    }
  }
  memcpy(dst_tensor->data_, src_tensor.data_, data_size);
  return RET_OK;
}

Tensor *Tensor::CopyTensor(const Tensor &src_tensor, bool copy_data) {
  auto *result = new (std::nothrow) Tensor;
  if (result == nullptr) {
    MS_LOG(ERROR) << "New tensor failed";
    return nullptr;
  }
  result->data_type_ = src_tensor.data_type_;
  result->shape_ = src_tensor.shape_;
  result->category_ = src_tensor.category_;
  result->format_ = src_tensor.format_;
  if (copy_data) {
    auto ret = CopyTensorData(src_tensor, result);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CopyTensorData error";
      return nullptr;
    }
  }
  return result;
}

Tensor::~Tensor() {
  if (nullptr != this->data_) {
    if (this->allocator_ != nullptr) {
      this->allocator_->Free(this->data_);
    } else {
      free(this->data_);
    }
    this->data_ = nullptr;
  }
}

bool Tensor::operator==(const Tensor &tensor) {
  return data_ == tensor.data_ && shape_ == tensor.shape_ && data_type_ == tensor.data_type_;
}

int32_t Tensor::Batch() const {
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return RET_ERROR;
  }
  switch (this->format_) {
    case schema::Format::Format_NHWC:
    case schema::Format::Format_NHWC4:
    case schema::Format::Format_NCHW:
    case schema::Format::Format_NC4HW4:
    case schema::Format::Format_KCHW:
    case schema::Format::Format_KHWC:
    case schema::Format::Format_NC:
    case schema::Format::Format_NC4:
      return this->shape_[0];
    case schema::Format::Format_HWCK:
    case schema::Format::Format_CHWK:
      return this->shape_[3];
    case schema::Format::Format_HWKC:
      return this->shape_[2];
    case schema::Format::Format_CKHW:
      return this->shape_[1];
    default:
      MS_LOG(ERROR) << "Unsupported format: " << EnumNameFormat(this->format_);
      return RET_ERROR;
  }
}

int32_t Tensor::Channel() const {
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return RET_ERROR;
  }
  switch (this->format_) {
    case schema::Format::Format_NCHW:
    case schema::Format::Format_KCHW:
    case schema::Format::Format_NC:
    case schema::Format::Format_NC4:
      return this->shape_[1];
    case schema::Format::Format_HWCK:
      return this->shape_[2];
    case schema::Format::Format_HWKC:
    case schema::Format::Format_NHWC:
    case schema::Format::Format_NHWC4:
    case schema::Format::Format_NC4HW4:
    case schema::Format::Format_KHWC:
      return this->shape_[3];
    case schema::Format::Format_CKHW:
    case schema::Format::Format_CHWK:
      return this->shape_[0];
    default:
      return RET_ERROR;
  }
}

int32_t Tensor::Height() const {
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return RET_ERROR;
  }
  switch (this->format_) {
    case schema::Format::Format_NCHW:
    case schema::Format::Format_KCHW:
    case schema::Format::Format_CKHW:
      return this->shape_[2];
    case schema::Format::Format_NHWC:
    case schema::Format::Format_NHWC4:
    case schema::Format::Format_NC4HW4:
    case schema::Format::Format_KHWC:
    case schema::Format::Format_CHWK:
      return this->shape_[1];
    case schema::Format::Format_HWCK:
    case schema::Format::Format_HWKC:
    case schema::Format::Format_HW:
    case schema::Format::Format_HW4:
      return this->shape_[0];
    default:
      MS_LOG(ERROR) << "Unsupported format: " << EnumNameFormat(this->format_);
      return RET_ERROR;
  }
}

int32_t Tensor::Width() const {
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return -1;
  }
  switch (this->format_) {
    case schema::Format::Format_NCHW:
    case schema::Format::Format_KCHW:
    case schema::Format::Format_CKHW:
      return this->shape_[3];
    case schema::Format::Format_KHWC:
    case schema::Format::Format_NHWC:
    case schema::Format::Format_NHWC4:
    case schema::Format::Format_NC4HW4:
    case schema::Format::Format_CHWK:
      return this->shape_[2];
    case schema::Format::Format_HWCK:
    case schema::Format::Format_HWKC:
    case schema::Format::Format_HW:
    case schema::Format::Format_HW4:
      return this->shape_[1];
    default:
      return RET_ERROR;
  }
}

size_t Tensor::Size() const {
  size_t element_size = DataTypeSize(this->data_type_);
  auto element_num = (format_ == schema::Format::Format_NC4HW4 || format_ == schema::Format::Format_NHWC4)
                       ? ElementsC4Num()
                       : ElementsNum();
  if (element_num < 0) {
    MS_LOG(ERROR) << "Element number of tensor should large than 0 : " << element_num;
    return 0;
  }
  return element_size * element_num;
}

int Tensor::ElementsNum() const {
  if (this->category_ == CONST_SCALAR) {
    return 1;
  }
  return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int>());
}

int32_t Tensor::ElementsC4Num() const {
  if (this->category_ == CONST_SCALAR) {
    return 1;
  }
  int32_t result = 1;
  if (this->shape_.size() == 4) {
    result = Batch() * Height() * Width() * ((Channel() + 3) / 4 * 4);
  } else if (this->shape_.size() == 2) {
    result = this->shape_[0] * ((this->shape_[1] + 3) / 4 * 4);
  }
  return result;
}

int Tensor::DimensionSize(const size_t index) const {
  int dim_size = -1;
  if (index < shape_.size()) {
    dim_size = shape_[index];
  } else {
    MS_LOG(ERROR) << "Dimension index is wrong: " << index;
  }
  return dim_size;
}

std::string Tensor::ToString() const {
  std::ostringstream oss;
  oss << "schema::Format: " << EnumNameFormat(this->format_);
  oss << " DataType: " << this->data_type_;
  oss << " Category: " << this->category_;
  oss << " Shape:";
  for (auto &dim : this->shape()) {
    oss << " " << dim;
  }
  oss << std::endl << "Data:";
  switch (this->data_type_) {
    case kNumberTypeFloat32: {
      oss << DataToString<float>(this->data_c(), this->ElementsNum());
    } break;
    case kNumberTypeFloat16: {
      oss << DataToString<int16_t>(this->data_c(), this->ElementsNum());
    } break;
    case kNumberTypeInt32: {
      oss << DataToString<int32_t>(this->data_c(), this->ElementsNum());
    } break;
    case kNumberTypeInt16: {
      oss << DataToString<int16_t>(this->data_c(), this->ElementsNum());
    } break;
    case kNumberTypeInt8: {
      oss << DataToString<int8_t>(this->data_c(), this->ElementsNum());
    } break;
    default:
      oss << "Unsupported data type to print";
      break;
  }
  return oss.str();
}

int Tensor::set_root_tensor(Tensor *tensor) {
  this->root_tensor_ = tensor;
  if (this->root_tensor_ == this) {
    return RET_OK;
  }
  if (this->root_tensor_ == nullptr) {
    return RET_OK;
  }
  this->shape_ = this->root_tensor_->shape_;
  this->format_ = this->root_tensor_->format_;
  this->data_type_ = this->root_tensor_->data_type_;
  this->category_ = this->root_tensor_->category_;
  this->quant_params_ = this->root_tensor_->quant_params_;
  this->quant_clusters_ = this->root_tensor_->quant_clusters_;
  return RET_OK;
}

int Tensor::MallocData(const mindspore::Allocator *allocator) {
  if (nullptr != this->data_) {
    return RET_OK;
  }
  if (allocator != nullptr) {
    allocator_ = const_cast<mindspore::Allocator *>(allocator);
  }
  if (allocator_ == nullptr) {
    this->data_ = malloc(this->Size());
  } else {
    this->data_ = allocator_->Malloc(this->Size());
  }
  if (nullptr == this->data_) {
    MS_LOG(ERROR) << "Malloc tensor data failed, size=" << this->Size();
    return RET_ERROR;
  }

  return RET_OK;
}

void Tensor::FreeData() {
  if (nullptr == this->data_) {
    return;
  }
  if (nullptr == allocator_) {
    free(this->data_);
    this->data_ = nullptr;
  } else {
    allocator_->Free(this->data_);
    this->data_ = nullptr;
  }
}

void *Tensor::MutableData() {
  if (this->root_tensor_ != nullptr) {
    if (this->root_tensor_ != this && this->root_tensor_->data_ == nullptr) {
      MS_LOG(ERROR) << "root tensor has not been malloced";
      return nullptr;
    } else if (this->root_tensor_ != this && this->root_tensor_->data_ != nullptr) {
      return this->root_tensor_->data_;
    } else {
      // malloc self
    }
  }
  if (this->data_ == nullptr) {
    auto ret = this->MallocData();
    if (ret != 0) {
      MS_LOG(WARNING) << "Malloc data failed";
    }
  }
  Prepare();
  return this->data_;
}

void Tensor::DecRefCount() {
  if (this->IsConst() || this->IsGraphInput()) {
    return;
  }
  this->ref_count_--;
  if (this->ref_count_ <= 0) {
    FreeData();
    this->ref_count_ = 0;
  }
}

void Tensor::AddQuantParam(const QuantArg &quant_arg) { this->quant_params_.push_back(quant_arg); }

std::vector<QuantArg> Tensor::quant_params() const { return this->quant_params_; }

std::vector<float> Tensor::quant_clusters() const { return this->quant_clusters_; }

void Tensor::set_quant_clusters(const std::vector<float> &clusters) { this->quant_clusters_ = clusters; }

bool Tensor::enable_huffman_code() const { return enable_huffman_code_; }

void Tensor::set_enable_huffman_code(bool enable_huffman_code) { this->enable_huffman_code_ = enable_huffman_code; }

std::vector<tensor::MSTensor *> TensorVectorCast(const std::vector<Tensor *> &src) {
  std::vector<tensor::MSTensor *> target(src.size());
  std::transform(src.begin(), src.end(), target.begin(), [](Tensor *t) { return dynamic_cast<tensor::MSTensor *>(t); });
  return target;
}

}  // namespace lite

tensor::MSTensor *tensor::MSTensor::CreateTensor(const std::string &name, TypeId type, const std::vector<int> &shape,
                                                 const void *data, size_t data_len) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor.";
    return nullptr;
  }
  tensor->set_data(const_cast<void *>(data));
  tensor->set_shape(shape);
  tensor->set_tensor_name(name);
  tensor->set_data_type(type);
  return tensor;
}

}  // namespace mindspore
