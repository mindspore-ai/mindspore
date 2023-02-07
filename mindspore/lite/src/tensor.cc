/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
    : category_(category) {
  tensor_c_ = {false, data_type, static_cast<int>(format), nullptr, shape.size()};
  if (shape.size() > MAX_SHAPE_SIZE) {
    tensor_c_.shape_size_ = 0;
    MS_LOG(WARNING) << "The shape-size has exceeded the limit 8, now is " << shape.size();
    return;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor_c_.shape_[i] = shape[i];
  }
}

int Tensor::CopyTensorData(const Tensor &src_tensor, Tensor *dst_tensor) {
  if (dst_tensor == nullptr) {
    MS_LOG(ERROR) << "dst_tensor is nullptr";
    return RET_PARAM_INVALID;
  }
  if (src_tensor.tensor_c_.data_ == nullptr) {
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
  (void)memcpy(dst_tensor->tensor_c_.data_, src_tensor.tensor_c_.data_, data_size);
  return RET_OK;
}

Tensor *Tensor::CopyTensor(const Tensor &src_tensor, bool copy_data, AllocatorPtr allocator) {
  auto *result = new (std::nothrow) Tensor;
  if (result == nullptr) {
    MS_LOG(ERROR) << "New tensor failed";
    return nullptr;
  }
  (void)memcpy(&result->tensor_c_, &src_tensor.tensor_c_, sizeof(TensorC));
  result->tensor_c_.data_ = nullptr;
  result->category_ = src_tensor.category_;
  result->compress_type_ = src_tensor.compress_type_;
  result->compressed_size_ = src_tensor.compressed_size_;
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

  for (const LiteQuantParam &quant : src_tensor.quant_params()) {
    result->AddQuantParam(quant);
  }

  return result;
}

Tensor::~Tensor() {
  FreeData();
  this->tensor_c_.data_ = nullptr;
}

bool Tensor::operator==(const Tensor &tensor) {
  return tensor_c_.data_ == tensor.tensor_c_.data_ && tensor_c_.shape_size_ == tensor.tensor_c_.shape_size_ &&
         tensor_c_.data_type_ == tensor.tensor_c_.data_type_ &&
         std::equal(tensor_c_.shape_, tensor_c_.shape_ + tensor_c_.shape_size_, tensor.tensor_c_.shape_);
}

int32_t Tensor::Batch() const {
  // Only 2D or 4D tensors have valid batch.
  if (this->tensor_c_.shape_size_ != C4NUM && this->tensor_c_.shape_size_ != C2NUM) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->tensor_c_.shape_size_;
    return RET_ERROR;
  }
  switch (this->tensor_c_.format_) {
    case mindspore::NHWC:
    case mindspore::NHWC4:
    case mindspore::NCHW:
    case mindspore::NC4HW4:
    case mindspore::NC8HW8:
    case mindspore::KCHW:
    case mindspore::KHWC:
    case mindspore::NC:
    case mindspore::NC4:
      return this->tensor_c_.shape_[0];
    case mindspore::HWCK:
    case mindspore::CHWK:
      if (this->tensor_c_.shape_size_ != C4NUM) {
        return RET_ERROR;
      }
      return this->tensor_c_.shape_[C3NUM];
    case mindspore::HWKC:
      if (this->tensor_c_.shape_size_ != C4NUM) {
        return RET_ERROR;
      }
      return this->tensor_c_.shape_[C2NUM];
    case mindspore::CKHW:
      return this->tensor_c_.shape_[1];
    default:
      MS_LOG(ERROR) << "Unsupported format: " << EnumNameFormat(static_cast<schema::Format>(this->tensor_c_.format_));
      return RET_ERROR;
  }
}

int32_t Tensor::Channel() const {
  // Only 2D or 4D tensors have valid channel.
  if (this->tensor_c_.shape_size_ != C4NUM && this->tensor_c_.shape_size_ != C2NUM) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->tensor_c_.shape_size_;
    return RET_ERROR;
  }
  switch (this->tensor_c_.format_) {
    case mindspore::NCHW:
    case mindspore::KCHW:
    case mindspore::NC:
    case mindspore::NC4:
    case mindspore::NC4HW4:
    case mindspore::NC8HW8:
      return this->tensor_c_.shape_[1];
    case mindspore::HWCK:
      if (this->tensor_c_.shape_size_ != C4NUM) {
        return RET_ERROR;
      }
      return this->tensor_c_.shape_[C2NUM];
    case mindspore::HWKC:
    case mindspore::NHWC:
    case mindspore::NHWC4:
    case mindspore::KHWC:
      if (this->tensor_c_.shape_size_ != C4NUM) {
        return RET_ERROR;
      }
      return this->tensor_c_.shape_[C3NUM];
    case mindspore::CKHW:
    case mindspore::CHWK:
      return this->tensor_c_.shape_[0];
    default:
      return RET_ERROR;
  }
}

int32_t Tensor::Height() const {
  // Only 2D or 4D tensors have valid height.
  if (this->tensor_c_.shape_size_ != C4NUM && this->tensor_c_.shape_size_ != C2NUM) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->tensor_c_.shape_size_;
    return RET_ERROR;
  }
  switch (this->tensor_c_.format_) {
    case mindspore::NCHW:
    case mindspore::KCHW:
    case mindspore::CKHW:
    case mindspore::NC4HW4:
    case mindspore::NC8HW8:
      if (this->tensor_c_.shape_size_ != C4NUM) {
        return RET_ERROR;
      }
      return this->tensor_c_.shape_[C2NUM];
    case mindspore::NHWC:
    case mindspore::NHWC4:
    case mindspore::KHWC:
    case mindspore::CHWK:
      return this->tensor_c_.shape_[1];
    case mindspore::HWCK:
    case mindspore::HWKC:
    case mindspore::HW:
    case mindspore::HW4:
      return this->tensor_c_.shape_[0];
    default:
      MS_LOG(ERROR) << "Unsupported format: " << EnumNameFormat(static_cast<schema::Format>(this->tensor_c_.format_));
      return RET_ERROR;
  }
}

int32_t Tensor::Width() const {
  // Only 2D or 4D tensors have valid width.
  if (this->tensor_c_.shape_size_ != C4NUM && this->tensor_c_.shape_size_ != C2NUM) {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->tensor_c_.shape_size_;
    return RET_ERROR;
  }
  switch (this->tensor_c_.format_) {
    case mindspore::NCHW:
    case mindspore::KCHW:
    case mindspore::CKHW:
    case mindspore::NC4HW4:
    case mindspore::NC8HW8:
      if (this->tensor_c_.shape_size_ != C4NUM) {
        return RET_ERROR;
      }
      return this->tensor_c_.shape_[C3NUM];
    case mindspore::KHWC:
    case mindspore::NHWC:
    case mindspore::NHWC4:
    case mindspore::CHWK:
      if (this->tensor_c_.shape_size_ != C4NUM) {
        return RET_ERROR;
      }
      return this->tensor_c_.shape_[C2NUM];
    case mindspore::HWCK:
    case mindspore::HWKC:
    case mindspore::HW:
    case mindspore::HW4:
      return this->tensor_c_.shape_[1];
    default:
      return RET_ERROR;
  }
}

size_t Tensor::Size() const {
  if (compress_type_ != kNoCompression) {
    return compressed_size_;
  } else {
    size_t element_size = DataTypeSize(static_cast<TypeId>(tensor_c_.data_type_));
    if (element_size == 0) {
      MS_LOG(INFO) << "Unexpected data type: " << tensor_c_.data_type_;
      return 0;
    }
    auto element_num = (tensor_c_.format_ == mindspore::NC4HW4 || tensor_c_.format_ == mindspore::NHWC4)
                         ? ElementsC4Num()
                         : ElementsNum();
    if (element_num <= 0) {
      std::vector<int> shape(tensor_c_.shape_, tensor_c_.shape_ + tensor_c_.shape_size_);
      MS_LOG(DEBUG) << "Element number of tensor should large than 0 : " << element_num << ", shape: " << shape;
      return 0;
    }
    return element_size * static_cast<size_t>(element_num);
  }
}

int64_t Tensor::ElementsNum() const {
  if (this->category_ == CONST_SCALAR) {
    return 1;
  }
  if (tensor_c_.format_ == mindspore::NC4HW4) {
    return ElementsC4Num();
  }
  if (tensor_c_.format_ == mindspore::NC8HW8) {
    return ElementsC8Num();
  }
  int64_t num = 1;
  for (size_t i = 0; i < tensor_c_.shape_size_; ++i) {
    if (tensor_c_.shape_[i] < 0) {
      return 0;
    }
    CHECK_INT64_MUL_OVERFLOW(num, tensor_c_.shape_[i]);
    num *= tensor_c_.shape_[i];
  }
  return num;
}

int64_t Tensor::ElementsC4Num() const {
  if (this->category_ == CONST_SCALAR) {
    return 1;
  }
  int64_t result = 1;
  constexpr int kC4Align = 4;
  if (this->tensor_c_.shape_size_ == C4NUM) {
    CHECK_INT64_MUL_OVERFLOW(result, Batch());
    result *= Batch();
    CHECK_INT64_MUL_OVERFLOW(result, Height());
    result *= Height();
    CHECK_INT64_MUL_OVERFLOW(result, Width());
    result *= Width();
    CHECK_INT64_MUL_OVERFLOW(result, (Channel() + 3LL) / kC4Align * kC4Align);
    result *= (Channel() + 3LL) / kC4Align * kC4Align;
  } else if (this->tensor_c_.shape_size_ == 3) {  // 3 : [H W C]
    CHECK_INT64_MUL_OVERFLOW(result, this->tensor_c_.shape_[0]);
    result *= this->tensor_c_.shape_[0];
    CHECK_INT64_MUL_OVERFLOW(result, this->tensor_c_.shape_[1]);
    result *= this->tensor_c_.shape_[1];
    CHECK_INT64_MUL_OVERFLOW(result, (this->tensor_c_.shape_[2] + 3LL) / kC4Align * kC4Align);  // C : 2
    result *= (this->tensor_c_.shape_[2] + 3LL) / kC4Align * kC4Align;                          // C : 2
  } else if (this->tensor_c_.shape_size_ == 2) {                                                // 2 : [W C]
    CHECK_INT64_MUL_OVERFLOW(result, this->tensor_c_.shape_[0]);
    result *= this->tensor_c_.shape_[0];
    CHECK_INT64_MUL_OVERFLOW(result, (this->tensor_c_.shape_[1] + 3LL) / kC4Align * kC4Align);
    result *= (this->tensor_c_.shape_[1] + 3LL) / kC4Align * kC4Align;
  } else if (this->tensor_c_.shape_size_ == 1) {  // 1 : C
    CHECK_INT64_MUL_OVERFLOW(result, (this->tensor_c_.shape_[0] + 3LL) / kC4Align * kC4Align);
    result *= (this->tensor_c_.shape_[0] + 3LL) / kC4Align * kC4Align;
  } else {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << this->shape().size();
  }
  return result;
}

int64_t Tensor::ElementsC8Num() const {
  if (this->category_ == CONST_SCALAR) {
    return 1;
  }
  int64_t result = 1;
  constexpr int kC8Align = 8;
  if (this->tensor_c_.shape_size_ == C4NUM) {
    CHECK_INT64_MUL_OVERFLOW(result, Batch());
    result *= Batch();
    CHECK_INT64_MUL_OVERFLOW(result, Height());
    result *= Height();
    CHECK_INT64_MUL_OVERFLOW(result, Width());
    result *= Width();
    CHECK_INT64_MUL_OVERFLOW(result, (Channel() + 7LL) / kC8Align * kC8Align);
    result *= (Channel() + 7LL) / kC8Align * kC8Align;
  } else if (this->tensor_c_.shape_size_ == C2NUM) {
    CHECK_INT64_MUL_OVERFLOW(result, this->tensor_c_.shape_[0]);
    result *= this->tensor_c_.shape_[0];
    CHECK_INT64_MUL_OVERFLOW(result, (this->tensor_c_.shape_[1] + 7LL) / kC8Align * kC8Align);
    result *= (this->tensor_c_.shape_[1] + 7LL) / kC8Align * kC8Align;
  }
  return result;
}

int Tensor::DimensionSize(const size_t index) const {
  int dim_size = -1;
  if (index < tensor_c_.shape_size_) {
    dim_size = tensor_c_.shape_[index];
  } else {
    MS_LOG(ERROR) << "Dimension index is wrong: " << index;
  }
  return dim_size;
}

std::string Tensor::ToString() const {
  std::ostringstream oss;
  oss << "Tensor name: " << this->tensor_name();
  oss << " schema::Format: " << EnumNameFormat(static_cast<schema::Format>(this->tensor_c_.format_));
  oss << " DataType: " << this->tensor_c_.data_type_;
  oss << " Category: " << this->category_;
  oss << " Shape:";
  for (auto &dim : this->shape()) {
    oss << " " << dim;
  }
  oss << std::endl << "Data:";
  auto data = tensor_c_.data_;
  switch (this->tensor_c_.data_type_) {
    case kNumberTypeFloat32: {
      oss << DataToString<float>(data, this->ElementsNum());
    } break;
    case kNumberTypeFloat16: {
      oss << DataToString<int16_t>(data, this->ElementsNum());
    } break;
    case kNumberTypeInt32: {
      oss << DataToString<int32_t>(data, this->ElementsNum());
    } break;
    case kNumberTypeInt16: {
      oss << DataToString<int16_t>(data, this->ElementsNum());
    } break;
    case kNumberTypeInt8: {
      oss << DataToString<int8_t>(data, this->ElementsNum());
    } break;
    default:
      oss << "Unsupported data type to print";
      break;
  }
  return oss.str();
}

int Tensor::MallocData(const AllocatorPtr allocator) {
  if (this->tensor_c_.data_ != nullptr) {
    return RET_OK;
  }
  if (allocator != nullptr) {
    allocator_ = allocator;
  }
  size_t element_size = DataTypeSize(static_cast<TypeId>(this->tensor_c_.data_type_));
  if (element_size == 0) {
    MS_LOG(ERROR) << "Unexpected data type: " << tensor_c_.data_type_;
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
    this->tensor_c_.data_ = malloc(data_size);
  } else {
    this->tensor_c_.data_ = allocator_->Malloc(data_size);
    allocator_->SetRefCount(this->tensor_c_.data_, 1);
  }
  if (this->tensor_c_.data_ == nullptr) {
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
  if (this->tensor_c_.data_ != nullptr && this->own_data_) {
    if (this->allocator_ != nullptr) {
      if (allocator_->DecRefCount(this->tensor_c_.data_, 1) <= 0) {
        allocator_->Free(this->tensor_c_.data_);  // Due to existing various allocator, here do not set data to nullptr.
      }
      if (!IS_STATIC_ALLOCATOR(allocator_) || allocator_->RefCount(this->tensor_c_.data_) != 0) {
        this->tensor_c_.data_ = nullptr;
      }
    } else {
      free(this->tensor_c_.data_);
      this->tensor_c_.data_ = nullptr;
    }
  } else if (this->category_ == Category::VAR) {
    if (!IS_STATIC_ALLOCATOR(allocator_) || allocator_->RefCount(this->tensor_c_.data_) != 0) {
      if (this->init_ref_count_ == 1) {
        this->tensor_c_.data_ = nullptr;
      }
    }
  }
}

void *Tensor::ReallocData() {
  if (this->tensor_c_.data_ != nullptr) {
    FreeData();
  }
  return this->MutableData();
}

void *Tensor::MutableData() {
  if (this->tensor_c_.data_ == nullptr) {
    auto ret = this->MallocData();
    if (ret != 0) {
      MS_LOG(WARNING) << "Malloc data failed";
    }
  }
  Prepare();
  return this->tensor_c_.data_;
}

void Tensor::DecRefCount() {
  if (this->IsGraphInput()) {
    return;
  }
  int tensor_ref_count = --ref_count_;
  if (tensor_ref_count <= 0) {
    tensor_c_.shape_changed_ = false;
    if (this->IsConst()) {
      return;
    }
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
  auto tensor = std::make_unique<lite::Tensor>();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor.";
    return nullptr;
  }

  size_t shape_size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      return nullptr;
    }
    shape_size *= static_cast<size_t>(shape[i]);
  }
  auto data_type_size = lite::DataTypeSize(type);
  if (data_type_size == 0) {
    MS_LOG(ERROR) << "not support create this type: " << type;
    return nullptr;
  }

  if (data == nullptr && data_len != 0) {
    MS_LOG(ERROR) << "shape, data type and data len not match.";
    return nullptr;
  }

  if (data != nullptr && data_len != shape_size * data_type_size) {
    MS_LOG(ERROR) << "shape, data type and data len not match.";
    return nullptr;
  }
  tensor->set_data(const_cast<void *>(data));
  tensor->set_shape(shape);
  tensor->set_tensor_name(name);
  tensor->set_data_type(type);
  tensor->set_category(data != nullptr ? (shape.empty() ? CONST_SCALAR : CONST_TENSOR) : VAR);
  return tensor.release();
}

Tensor *Tensor::CreateTensorByDeepCopy(const std::string &name, TypeId type, const std::vector<int> &shape,
                                       const void *data, size_t data_len) {
  auto tensor = std::make_unique<lite::Tensor>();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor.";
    return nullptr;
  }

  auto data_type_size = lite::DataTypeSize(type);
  if (data_type_size == 0) {
    MS_LOG(ERROR) << "not support create this type: " << type;
    return nullptr;
  }

  if (data_len > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "data length is invalid.";
    return nullptr;
  } else if (data_len == 0 && data != nullptr) {
    MS_LOG(ERROR) << "data length and data are not match.";
    return nullptr;
  } else if (data_len == 0 && data == nullptr) {
    tensor->set_data(const_cast<void *>(data));
  } else {
    void *new_data = malloc(data_len);
    if (new_data == nullptr) {
      MS_LOG(ERROR) << "Failed to malloc data.";
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
  tensor->set_category(data != nullptr ? (shape.empty() ? CONST_SCALAR : CONST_TENSOR) : VAR);
  return tensor.release();
}

}  // namespace lite

}  // namespace mindspore
