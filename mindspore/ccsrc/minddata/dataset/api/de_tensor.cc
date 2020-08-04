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

#include "minddata/dataset/include/de_tensor.h"
#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "utils/hashing.h"
#include "mindspore/lite/src/ir/tensor.h"

namespace mindspore {
namespace tensor {
dataset::DataType MSTypeToDEType(TypeId data_type) {
  switch (data_type) {
    case kNumberTypeBool:
      return dataset::DataType(dataset::DataType::DE_BOOL);
    case kNumberTypeInt8:
      return dataset::DataType(dataset::DataType::DE_INT8);
    case kNumberTypeUInt8:
      return dataset::DataType(dataset::DataType::DE_UINT8);
    case kNumberTypeInt16:
      return dataset::DataType(dataset::DataType::DE_INT16);
    case kNumberTypeUInt16:
      return dataset::DataType(dataset::DataType::DE_UINT16);
    case kNumberTypeInt32:
      return dataset::DataType(dataset::DataType::DE_INT32);
    case kNumberTypeUInt32:
      return dataset::DataType(dataset::DataType::DE_UINT32);
    case kNumberTypeInt64:
      return dataset::DataType(dataset::DataType::DE_INT64);
    case kNumberTypeUInt64:
      return dataset::DataType(dataset::DataType::DE_UINT64);
    case kNumberTypeFloat16:
      return dataset::DataType(dataset::DataType::DE_FLOAT16);
    case kNumberTypeFloat32:
      return dataset::DataType(dataset::DataType::DE_FLOAT32);
    case kNumberTypeFloat64:
      return dataset::DataType(dataset::DataType::DE_FLOAT64);
    default:
      return dataset::DataType(dataset::DataType::DE_UNKNOWN);
  }
}

TypeId DETypeToMSType(dataset::DataType data_type) {
  switch (data_type.value()) {
    case dataset::DataType::DE_BOOL:
      return mindspore::TypeId::kNumberTypeBool;
    case dataset::DataType::DE_INT8:
      return mindspore::TypeId::kNumberTypeInt8;
    case dataset::DataType::DE_UINT8:
      return mindspore::TypeId::kNumberTypeUInt8;
    case dataset::DataType::DE_INT16:
      return mindspore::TypeId::kNumberTypeInt16;
    case dataset::DataType::DE_UINT16:
      return mindspore::TypeId::kNumberTypeUInt16;
    case dataset::DataType::DE_INT32:
      return mindspore::TypeId::kNumberTypeInt32;
    case dataset::DataType::DE_UINT32:
      return mindspore::TypeId::kNumberTypeUInt32;
    case dataset::DataType::DE_INT64:
      return mindspore::TypeId::kNumberTypeInt64;
    case dataset::DataType::DE_UINT64:
      return mindspore::TypeId::kNumberTypeUInt64;
    case dataset::DataType::DE_FLOAT16:
      return mindspore::TypeId::kNumberTypeFloat16;
    case dataset::DataType::DE_FLOAT32:
      return mindspore::TypeId::kNumberTypeFloat32;
    case dataset::DataType::DE_FLOAT64:
      return mindspore::TypeId::kNumberTypeFloat64;
    default:
      return kTypeUnknown;
  }
}

MSTensor *DETensor::CreateTensor(TypeId data_type, const std::vector<int> &shape) {
  return new DETensor(data_type, shape);
}

MSTensor *DETensor::CreateTensor(const std::string &path) {
  std::shared_ptr<dataset::Tensor> t;
  (void)dataset::Tensor::CreateFromFile(path, &t);
  return new DETensor(std::move(t));
}

DETensor::DETensor(TypeId data_type, const std::vector<int> &shape) {
  std::vector<dataset::dsize_t> t_shape;
  t_shape.reserve(shape.size());
  std::transform(shape.begin(), shape.end(), std::back_inserter(t_shape),
                 [](int s) -> dataset::dsize_t { return static_cast<dataset::dsize_t>(s); });
  dataset::Tensor::CreateEmpty(dataset::TensorShape(t_shape), MSTypeToDEType(data_type), &this->tensor_impl_);
}

DETensor::DETensor(std::shared_ptr<dataset::Tensor> tensor_ptr) { this->tensor_impl_ = std::move(tensor_ptr); }

MSTensor *DETensor::ConvertToLiteTensor() {
  // static MSTensor::CreateTensor is only for the LiteTensor
  MSTensor *tensor = MSTensor::CreateTensor(this->data_type(), this->shape());
  MS_ASSERT(tensor->Size() == this->Size());
  memcpy_s(tensor->MutableData(), tensor->Size(), this->MutableData(), this->Size());
  return tensor;
}

std::shared_ptr<dataset::Tensor> DETensor::tensor() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_;
}

TypeId DETensor::data_type() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return DETypeToMSType(this->tensor_impl_->type());
}

TypeId DETensor::set_data_type(TypeId data_type) {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  if (data_type != this->data_type()) {
    std::shared_ptr<dataset::Tensor> temp;
    dataset::Tensor::CreateFromMemory(this->tensor_impl_->shape(), MSTypeToDEType(data_type),
                                      this->tensor_impl_->GetBuffer(), &temp);
    this->tensor_impl_ = temp;
  }
  return data_type;
}

std::vector<int> DETensor::shape() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  std::vector<dataset::dsize_t> t_shape = this->tensor_impl_->shape().AsVector();
  std::vector<int> shape;
  shape.reserve(t_shape.size());
  std::transform(t_shape.begin(), t_shape.end(), std::back_inserter(shape),
                 [](dataset::dsize_t s) -> int { return static_cast<int>(s); });
  return shape;
}

size_t DETensor::set_shape(const std::vector<int> &shape) {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  std::vector<dataset::dsize_t> t_shape;
  t_shape.reserve(shape.size());
  std::transform(shape.begin(), shape.end(), std::back_inserter(t_shape),
                 [](int s) -> dataset::dsize_t { return static_cast<dataset::dsize_t>(s); });
  dataset::Status rc = this->tensor_impl_->Reshape(dataset::TensorShape(t_shape));
  return shape.size();
}

int DETensor::DimensionSize(size_t index) const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  int dim_size = -1;
  auto shape = this->shape();
  if (index < shape.size()) {
    dim_size = shape[index];
  } else {
    MS_LOG(ERROR) << "Dimension index is wrong: " << index;
  }
  return dim_size;
}

int DETensor::ElementsNum() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->Size();
}

std::size_t DETensor::hash() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  auto shape = this->shape();
  std::size_t hash_value = std::hash<int>{}(SizeToInt(this->data_type()));
  hash_value = hash_combine(hash_value, std::hash<size_t>{}(shape.size()));
  // hash all elements may costly, so only take at most 4 elements into account based on
  // some experiments.
  for (size_t i = 0; (i < shape.size()) && (i < 4); ++i) {
    hash_value = hash_combine(hash_value, (std::hash<int>{}(shape[i])));
  }
  return hash_value;
}

size_t DETensor::Size() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->SizeInBytes();
}

void *DETensor::MutableData() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->GetMutableBuffer();
}

}  // namespace tensor
}  // namespace mindspore
