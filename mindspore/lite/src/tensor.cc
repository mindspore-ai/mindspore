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

#include "src/tensor.h"
#include <vector>
#include <string>
#include <utility>
#include "schema/ops_types_generated.h"
#include "securec/include/securec.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
namespace {
static const size_t max_malloc_size_ = GetMaxMallocSize();
}
#if ENABLE_HIGH_PERFORMANCE
#define CHECK_INT64_MUL_OVERFLOW(x, y)
#else
#define CHECK_INT64_MUL_OVERFLOW(x, y)       \
  do {                                       \
    if (INT64_MUL_OVERFLOW(x, y)) {          \
      MS_LOG(ERROR) << "INT64 MUL OVERFLOW"; \
      return INT64_MAX;                      \
    }                                        \
  } while (0)

#define INT64_MUL_OVERFLOW(x, y)                                                                   \
  (((x) == 0) ? false                                                                              \
              : ((x) > 0 ? (((y) >= 0) ? (INT64_MAX / (x)) < (y) : (INT64_MAX / (x)) < (-1 * (y))) \
                         : (((y) >= 0) ? (INT64_MAX / (x)) > (-1 * (y)) : (INT64_MAX / (x)) > (y))))
#endif

Tensor::Tensor(const TypeId data_type, std::vector<int> shape, const mindspore::Format &format, Category category)
    : data_type_(data_type), shape_(std::move(shape)), format_(format), category_(category) {}

int Tensor::CopyTensorData(const Tensor &src_tensor, Tensor *dst_tensor) {
  if (dst_tensor == nullptr) {
    MS_LOG(ERROR) << "dst_tensor is nullptr";
    return RET_PARAM_INVALID;
  }
  if (src_tensor.data_ == nullptr) {
    MS_LOG(INFO) << "data of src tensor is nullptr";
    return RET_OK;
  }
  size_t data_size = dst_tensor->Size();
  if (data_size != src_tensor.Size()) {
    MS_LOG(ERROR) << "Size of dst tensor is not compatible with src tensor";
    return RET_ERROR;
  }
  if (dst_tensor->MallocData() != RET_OK) {
    MS_LOG(ERROR) << "Malloc memory failed";
    return RET_ERROR;
  }
  dst_tensor->ResetRefCount();
  memcpy(dst_tensor->data_, src_tensor.data_, data_size);
  return RET_OK;
}

Tensor *Tensor::CopyTensor(const Tensor &src_tensor, bool copy_data, AllocatorPtr allocator) {
  auto *result = new (std::nothrow) Tensor;
  if (result == nullptr) {
    MS_LOG(ERROR) << "New tensor failed";
    return nullptr;
  }
  result->data_type_ = src_tensor.data_type_;
  result->shape_ = src_tensor.shape_;
  result->category_ = src_tensor.category_;
  result->format_ = src_tensor.format_;
  result->set_allocator(allocator);
  result->set_tensor_name(src_tensor.tensor_name() + "_duplicate");
  if (copy_data) {
    auto ret = CopyTensorData(src_tensor, result);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CopyTensorData error";
      delete result;
      return nullptr;
    }
    result->own_data_ = src_tensor.own_data_;
  }

  for (LiteQuantParam quant : src_tensor.quant_params()) {
    result->AddQuantParam(quant);
  }

  return result;
}

Tensor::~Tensor() {
  FreeData();
  this->data_ = nullptr;
}

bool Tensor::operator==(const Tensor &tensor) {
  return data_ == tensor.data_ && shape_ == tensor.shape_ && data_type_ == tensor.data_type_;
}

int32_t Tensor::Batch() const {
  // Only 2D or 4D tensors have valid batch.
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return RET_ERROR;
  }
  switch (this->format_) {
    case mindspore::NHWC:
    case mindspore::NHWC4:
    case mindspore::NCHW:
    case mindspore::NC4HW4:
    case mindspore::NC8HW8:
    case mindspore::KCHW:
    case mindspore::KHWC:
    case mindspore::NC:
    case mindspore::NC4:
      return this->shape_[0];
    case mindspore::HWCK:
    case mindspore::CHWK:
      if (this->shape_.size() != 4) {
        return RET_ERROR;
      }
      return this->shape_[3];
    case mindspore::HWKC:
      if (this->shape_.size() != 4) {
        return RET_ERROR;
      }
      return this->shape_[2];
    case mindspore::CKHW:
      return this->shape_[1];
    default:
      MS_LOG(ERROR) << "Unsupported format: " << EnumNameFormat(static_cast<schema::Format>(this->format_));
      return RET_ERROR;
  }
}

int32_t Tensor::Channel() const {
  // Only 2D or 4D tensors have valid channel.
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return RET_ERROR;
  }
  switch (this->format_) {
    case mindspore::NCHW:
    case mindspore::KCHW:
    case mindspore::NC:
    case mindspore::NC4:
    case mindspore::NC4HW4:
    case mindspore::NC8HW8:
      return this->shape_[1];
    case mindspore::HWCK:
      if (this->shape_.size() != 4) {
        return RET_ERROR;
      }
      return this->shape_[2];
    case mindspore::HWKC:
    case mindspore::NHWC:
    case mindspore::NHWC4:
    case mindspore::KHWC:
      if (this->shape_.size() != 4) {
        return RET_ERROR;
      }
      return this->shape_[3];
    case mindspore::CKHW:
    case mindspore::CHWK:
      return this->shape_[0];
    default:
      return RET_ERROR;
  }
}

int32_t Tensor::Height() const {
  // Only 2D or 4D tensors have valid height.
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return RET_ERROR;
  }
  switch (this->format_) {
    case mindspore::NCHW:
    case mindspore::KCHW:
    case mindspore::CKHW:
    case mindspore::NC4HW4:
    case mindspore::NC8HW8:
      if (this->shape_.size() != 4) {
        return RET_ERROR;
      }
      return this->shape_[2];
    case mindspore::NHWC:
    case mindspore::NHWC4:
    case mindspore::KHWC:
    case mindspore::CHWK:
      return this->shape_[1];
    case mindspore::HWCK:
    case mindspore::HWKC:
    case mindspore::HW:
    case mindspore::HW4:
      return this->shape_[0];
    default:
      MS_LOG(ERROR) << "Unsupported format: " << EnumNameFormat(static_cast<schema::Format>(this->format_));
      return RET_ERROR;
  }
}

int32_t Tensor::Width() const {
  // Only 2D or 4D tensors have valid width.
  if (this->shape_.size() != 4 && this->shape_.size() != 2) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
    return RET_ERROR;
  }
  switch (this->format_) {
    case mindspore::NCHW:
    case mindspore::KCHW:
    case mindspore::CKHW:
    case mindspore::NC4HW4:
    case mindspore::NC8HW8:
      if (this->shape_.size() != 4) {
        return RET_ERROR;
      }
      return this->shape_[3];
    case mindspore::KHWC:
    case mindspore::NHWC:
    case mindspore::NHWC4:
    case mindspore::CHWK:
      if (this->shape_.size() != 4) {
        return RET_ERROR;
      }
      return this->shape_[2];
    case mindspore::HWCK:
    case mindspore::HWKC:
    case mindspore::HW:
    case mindspore::HW4:
      return this->shape_[1];
    default:
      return RET_ERROR;
  }
}

size_t Tensor::Size() const {
  size_t element_size = DataTypeSize(this->data_type_);
  if (element_size == 0) {
    MS_LOG(INFO) << "Unexpected data type: " << data_type_;
    return 0;
  }
  auto element_num = (format_ == mindspore::NC4HW4 || format_ == mindspore::NHWC4) ? ElementsC4Num() : ElementsNum();
  if (element_num <= 0) {
    MS_LOG(DEBUG) << "Element number of tensor should large than 0 : " << element_num << ", shape: " << shape_;
    return 0;
  }
  return element_size * static_cast<size_t>(element_num);
}

int64_t Tensor::ElementsNum() const {
  if (this->category_ == CONST_SCALAR) {
    return 1;
  }
  if (format_ == mindspore::NC4HW4) {
    return ElementsC4Num();
  }
  if (format_ == mindspore::NC8HW8) {
    return ElementsC8Num();
  }
  int64_t num = 1;
  for (size_t i = 0; i < shape_.size(); ++i) {
    CHECK_INT64_MUL_OVERFLOW(num, shape_[i]);
    num *= shape_[i];
  }
  return num;
}

int64_t Tensor::ElementsC4Num() const {
  if (this->category_ == CONST_SCALAR) {
    return 1;
  }
  int64_t result = 1;
  constexpr int kC4Align = 4;
  if (this->shape_.size() == 4) {
    CHECK_INT64_MUL_OVERFLOW(result, Batch());
    result *= Batch();
    CHECK_INT64_MUL_OVERFLOW(result, Height());
    result *= Height();
    CHECK_INT64_MUL_OVERFLOW(result, Width());
    result *= Width();
    CHECK_INT64_MUL_OVERFLOW(result, (Channel() + 3LL) / kC4Align * kC4Align);
    result *= (Channel() + 3LL) / kC4Align * kC4Align;
  } else if (this->shape_.size() == 2) {
    CHECK_INT64_MUL_OVERFLOW(result, this->shape_[0]);
    result *= this->shape_[0];
    CHECK_INT64_MUL_OVERFLOW(result, (this->shape_[1] + 3LL) / kC4Align * kC4Align);
    result *= (this->shape_[1] + 3LL) / kC4Align * kC4Align;
  }
  return result;
}

int64_t Tensor::ElementsC8Num() const {
  if (this->category_ == CONST_SCALAR) {
    return 1;
  }
  int64_t result = 1;
  constexpr int kC8Align = 8;
  if (this->shape_.size() == 4) {
    CHECK_INT64_MUL_OVERFLOW(result, Batch());
    result *= Batch();
    CHECK_INT64_MUL_OVERFLOW(result, Height());
    result *= Height();
    CHECK_INT64_MUL_OVERFLOW(result, Width());
    result *= Width();
    CHECK_INT64_MUL_OVERFLOW(result, (Channel() + 7LL) / kC8Align * kC8Align);
    result *= (Channel() + 7LL) / kC8Align * kC8Align;
  } else if (this->shape_.size() == 2) {
    CHECK_INT64_MUL_OVERFLOW(result, this->shape_[0]);
    result *= this->shape_[0];
    CHECK_INT64_MUL_OVERFLOW(result, (this->shape_[1] + 7LL) / kC8Align * kC8Align);
    result *= (this->shape_[1] + 7LL) / kC8Align * kC8Align;
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
  oss << "Tensor name: " << this->tensor_name();
  oss << " schema::Format: " << EnumNameFormat(static_cast<schema::Format>(this->format_));
  oss << " DataType: " << this->data_type_;
  oss << " Category: " << this->category_;
  oss << " Shape:";
  for (auto &dim : this->shape()) {
    oss << " " << dim;
  }
  oss << std::endl << "Data:";
  switch (this->data_type_) {
    case kNumberTypeFloat32: {
      oss << DataToString<float>(data_, this->ElementsNum());
    } break;
    case kNumberTypeFloat16: {
      oss << DataToString<int16_t>(data_, this->ElementsNum());
    } break;
    case kNumberTypeInt32: {
      oss << DataToString<int32_t>(data_, this->ElementsNum());
    } break;
    case kNumberTypeInt16: {
      oss << DataToString<int16_t>(data_, this->ElementsNum());
    } break;
    case kNumberTypeInt8: {
      oss << DataToString<int8_t>(data_, this->ElementsNum());
    } break;
    default:
      oss << "Unsupported data type to print";
      break;
  }
  return oss.str();
}

int Tensor::MallocData(const AllocatorPtr allocator) {
  if (this->data_ != nullptr) {
    return RET_OK;
  }
  if (allocator != nullptr) {
    allocator_ = allocator;
  }
  size_t element_size = DataTypeSize(this->data_type_);
  if (element_size == 0) {
    MS_LOG(ERROR) << "Unexpected data type: " << data_type_;
    return RET_ERROR;
  }
  auto data_size = this->Size();
  if (data_size <= 0) {
    MS_LOG(INFO) << "Data size=" << data_size << " bytes";
    // expect return, currently not return for case (0,xx) shape tensor (where_fp32)
  }
  if (data_size > max_malloc_size_) {
    MS_LOG(ERROR) << "Malloc size is too big while coping data, " << data_size << " bytes";
    return RET_ERROR;
  }
  if (allocator_ == nullptr) {
    this->data_ = malloc(data_size);
  } else {
    this->data_ = allocator_->Malloc(data_size);
    allocator_->SetRefCount(this->data_, 1);
  }
  if (this->data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc tensor data failed, size=" << data_size;
    return RET_ERROR;
  }
  this->own_data_ = true;
  return RET_OK;
}

void Tensor::FreeData() {
  if (IS_RUNTIME_ALLOCATOR(allocator_)) {
    return;
  }
  if (this->data_ != nullptr && this->own_data_) {
    if (this->allocator_ != nullptr) {
      if (allocator_->DecRefCount(this->data_, 1) <= 0) {
        allocator_->Free(this->data_);  // Due to existing various allocator, here do not set data to nullptr.
      }
      if (!IS_STATIC_ALLOCATOR(allocator_) || allocator_->RefCount(this->data_) != 0) {
        this->data_ = nullptr;
      }
    } else {
      free(this->data_);
      this->data_ = nullptr;
    }
  }
}

void *Tensor::ReallocData() {
  if (this->data_ != nullptr) {
    FreeData();
  }
  return this->MutableData();
}

void *Tensor::MutableData() {
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
  int tensor_ref_count = --ref_count_;
  if (tensor_ref_count <= 0) {
    FreeData();
  }
}

void Tensor::AddQuantParam(const LiteQuantParam &quant_param) { this->quant_params_.push_back(quant_param); }

void Tensor::ClearQuantParam() {
  this->quant_params().clear();
  std::vector<LiteQuantParam>().swap(quant_params_);
}

std::vector<LiteQuantParam> Tensor::quant_params() const { return this->quant_params_; }

void Tensor::set_quant_params(const std::vector<LiteQuantParam> quant_params) { this->quant_params_ = quant_params; }

std::vector<float> Tensor::quant_clusters() const { return this->quant_clusters_; }

void Tensor::set_quant_clusters(const std::vector<float> &clusters) { this->quant_clusters_ = clusters; }

Tensor *Tensor::CreateTensor(const std::string &name, TypeId type, const std::vector<int> &shape, const void *data,
                             size_t data_len) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor.";
    return nullptr;
  }

  size_t shape_size = 1;
  if (shape.empty()) {
    shape_size = 0;
  } else {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        delete tensor;
        return nullptr;
      }
      shape_size *= static_cast<size_t>(shape[i]);
    }
  }
  auto data_type_size = lite::DataTypeSize(type);
  if (data_type_size == 0) {
    MS_LOG(ERROR) << "not support create this type: " << type;
    delete tensor;
    return nullptr;
  }

  if (data == nullptr && data_len != 0) {
    MS_LOG(ERROR) << "shape, data type and data len not match.";
    delete tensor;
    return nullptr;
  }

  if (data != nullptr && data_len != shape_size * data_type_size) {
    MS_LOG(ERROR) << "shape, data type and data len not match.";
    delete tensor;
    return nullptr;
  }
  tensor->set_data(const_cast<void *>(data));
  tensor->set_shape(shape);
  tensor->set_tensor_name(name);
  tensor->set_data_type(type);
  return tensor;
}

Tensor *Tensor::CreateTensorByDeepCopy(const std::string &name, TypeId type, const std::vector<int> &shape,
                                       const void *data, size_t data_len) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor.";
    return nullptr;
  }

  auto data_type_size = lite::DataTypeSize(type);
  if (data_type_size == 0) {
    MS_LOG(ERROR) << "not support create this type: " << type;
    delete tensor;
    return nullptr;
  }

  if (data_len < 0 || data_len > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "data length is invalid.";
    delete tensor;
    return nullptr;
  } else if (data_len == 0 && data != nullptr) {
    MS_LOG(ERROR) << "data length and data are not match.";
    delete tensor;
    return nullptr;
  } else if (data_len == 0 && data == nullptr) {
    tensor->set_data(const_cast<void *>(data));
  } else {
    void *new_data = malloc(data_len);
    if (new_data == nullptr) {
      MS_LOG(ERROR) << "Failed to malloc data.";
      delete tensor;
      return nullptr;
    }
    if (data != nullptr) {
      (void)memcpy(new_data, data, data_len);
    }
    tensor->set_data(const_cast<void *>(new_data));
  }

  size_t shape_size = 1;
  if (shape.empty()) {
    shape_size = 0;
  } else {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0) {
        delete tensor;
        return nullptr;
      }
      shape_size *= static_cast<size_t>(shape[i]);
    }
  }
  if (data_len != shape_size * data_type_size) {
    std::vector<int> truncate_shape = {static_cast<int>(data_len)};
    tensor->set_shape(truncate_shape);
  } else {
    tensor->set_shape(shape);
  }
  tensor->set_tensor_name(name);
  tensor->set_data_type(type);
  return tensor;
}

}  // namespace lite

}  // namespace mindspore
