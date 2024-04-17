/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "ir/dtype/tensor_type.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
TypePtr UndeterminedType::DeepCopy() const {
  if (IsGeneric() || element_type_ == nullptr) {
    return std::make_shared<UndeterminedType>();
  }
  return std::make_shared<UndeterminedType>(element_type_->DeepCopy());
}

std::string UndeterminedType::ToReprString() const {
  if (element_type_ == nullptr) {
    return "Undetermined";
  }
  return "Undetermined[" + element_type_->ToReprString() + "]";
}

std::string UndeterminedType::ToString() const {
  if (element_type_ == nullptr) {
    return "Undetermined";
  }
  return "Undetermined[" + element_type_->ToString() + "]";
}

std::string UndeterminedType::DumpText() const {
  if (element_type_ == nullptr) {
    return "Undetermined";
  }
  return "Undetermined[" + element_type_->DumpText() + "]";
}

bool UndeterminedType::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  const auto &other_type = static_cast<const UndeterminedType &>(other);
  return common::IsEqual(element_type_, other_type.element_type_);
}

size_t UndeterminedType::hash() const {
  size_t hash_value = hash_combine(static_cast<size_t>(kMetaTypeObject), static_cast<size_t>(object_type()));
  if (element_type_ != nullptr) {
    hash_value = hash_combine(hash_value, element_type_->hash());
  }
  return hash_value;
}

TypePtr TensorType::DeepCopy() const {
  if (element_type_ == nullptr) {
    return std::make_shared<TensorType>();
  }
  if (IsGeneric()) {
    return std::make_shared<TensorType>();
  }
  return std::make_shared<TensorType>(element_type_->DeepCopy());
}

std::string TensorType::ToReprString() const {
  if (element_type_ == nullptr) {
    return "tensor";
  }
  return "tensor[" + element_type_->ToReprString() + "]";
}

std::string TensorType::ToString() const {
  if (element_type_ == nullptr) {
    return "Tensor";
  }
  return "Tensor[" + element_type_->ToString() + "]";
}

std::string TensorType::DumpText() const {
  if (element_type_ == nullptr) {
    return "Tensor";
  }
  return "Tensor(" + element_type_->DumpText() + ")";
}

bool TensorType::operator==(const Type &other) const {
  if (this == &other) {
    return true;
  }
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  return *this == static_cast<const TensorType &>(other);
}

bool TensorType::operator==(const TensorType &other) const {
  if (other.isa<AnyType>()) {
    return false;
  }
  return common::IsEqual(element_type_, other.element_type_);
}

size_t TensorType::hash() const {
  size_t hash_value = hash_combine(static_cast<size_t>(kMetaTypeObject), static_cast<size_t>(object_type()));
  if (element_type_ != nullptr) {
    hash_value = hash_combine(hash_value, element_type_->hash());
  }
  return hash_value;
}

std::string AnyType::ToString() const {
  if (element() == nullptr) {
    return "Any(Tensor)";
  }
  return "Any(Tensor)[" + element()->ToString() + "]";
}

std::string AnyType::DumpText() const {
  if (element() == nullptr) {
    return "Any(Tensor)";
  }
  return "Any(Tensor)(" + element()->DumpText() + ")";
}

bool AnyType::operator==(const Type &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AnyType>()) {
    return false;
  }
  return *this == static_cast<const AnyType &>(other);
}

bool AnyType::operator==(const AnyType &other) const {
  if (this == &other) {
    return true;
  }
  return common::IsEqual(element(), other.element());
}

std::string NegligibleType::ToString() const {
  if (element() == nullptr) {
    return "Negligible(Tensor)";
  }
  return "Negligible(Tensor)[" + element()->ToString() + "]";
}

std::string NegligibleType::DumpText() const {
  if (element() == nullptr) {
    return "Negligible(Tensor)";
  }
  return "Negligible(Tensor)(" + element()->DumpText() + ")";
}

std::string SparseTensorType::ElementsDtypeStr(const StringType str_type) const {
  std::ostringstream oss;
  for (const TypePtr &elem : elements_) {
    if (str_type == kToString) {
      oss << elem->ToString();
    } else if (str_type == kDumpText) {
      oss << elem->DumpText();
    } else if (str_type == kReprString) {
      oss << elem->ToReprString();
    }
    oss << ",";
  }
  return oss.str();
}

std::string SparseTensorType::ToString() const {
  if (elements_.empty()) {
    return GetSparseTensorTypeName();
  }
  return GetSparseTensorTypeName() + "[" + ElementsDtypeStr(kToString) + "]";
}

std::string SparseTensorType::DumpText() const {
  if (elements_.empty()) {
    return GetSparseTensorTypeName();
  }
  return GetSparseTensorTypeName() + "[" + ElementsDtypeStr(kDumpText) + "]";
}

std::string SparseTensorType::ToReprString() const {
  if (elements_.empty()) {
    return GetSparseTensorTypeName();
  }
  return GetSparseTensorTypeName() + "[" + ElementsDtypeStr(kReprString) + "]";
}

const TypePtrList SparseTensorType::ElementsClone() const {
  TypePtrList elems;
  (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(elems), [](const TypePtr &ele) {
    MS_EXCEPTION_IF_NULL(ele);
    return ele->DeepCopy();
  });
  return elems;
}

TypePtr SparseTensorType::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<SparseTensorType>();
  }
  return std::make_shared<SparseTensorType>(ElementsClone());
}

const TypePtr SparseTensorType::operator[](std::size_t dim) const {
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "Index " << dim << " is out range of the SparseTensorType size " << size() << ".";
  }
  return elements_[dim];
}

bool SparseTensorType::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  const auto &other_type = static_cast<const SparseTensorType &>(other);
  return TypeListEqual()(elements_, other_type.elements_);
}

size_t SparseTensorType::hash() const {
  size_t hash_value = hash_combine(static_cast<size_t>(kMetaTypeObject), static_cast<size_t>(object_type()));
  return hash_combine(hash_value, TypeListHasher()(elements_));
}

TypePtr RowTensorType::DeepCopy() const {
  MS_EXCEPTION_IF_NULL(element_type_);
  if (IsGeneric()) {
    return std::make_shared<RowTensorType>();
  }
  return std::make_shared<RowTensorType>(element_type_->DeepCopy());
}

std::string RowTensorType::ToReprString() const {
  if (element_type_ == nullptr) {
    return "RowTensor";
  }
  return "RowTensor[" + element_type_->ToReprString() + "]";
}

std::string RowTensorType::ToString() const {
  if (element_type_ == nullptr) {
    return "RowTensor";
  }
  return "RowTensor[" + element_type_->ToString() + "]";
}

std::string RowTensorType::DumpText() const {
  if (element_type_ == nullptr) {
    return "RowTensor";
  }
  return "RowTensor[" + element_type_->DumpText() + "]";
}

bool RowTensorType::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  const auto &other_type = static_cast<const RowTensorType &>(other);
  return common::IsEqual(element_type_, other_type.element_type_);
}

size_t RowTensorType::hash() const {
  size_t hash_value = hash_combine(static_cast<size_t>(kMetaTypeObject), static_cast<size_t>(object_type()));
  if (element_type_ != nullptr) {
    hash_value = hash_combine(hash_value, element_type_->hash());
  }
  return hash_value;
}

TypePtr COOTensorType::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<COOTensorType>();
  }
  return std::make_shared<COOTensorType>(ElementsClone());
}

TypePtr CSRTensorType::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<CSRTensorType>();
  }
  return std::make_shared<CSRTensorType>(ElementsClone());
}

TypePtr MapTensorType::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<MapTensorType>();
  }
  MS_EXCEPTION_IF_NULL(key_dtype_);
  MS_EXCEPTION_IF_NULL(value_dtype_);
  return std::make_shared<MapTensorType>(key_dtype_->DeepCopy(), value_dtype_->DeepCopy());
}

std::string MapTensorType::ToString() const {
  if (IsGeneric()) {
    return "MapTensor";
  }
  MS_EXCEPTION_IF_NULL(key_dtype_);
  MS_EXCEPTION_IF_NULL(value_dtype_);
  return "MapTensor[" + key_dtype_->ToString() + ", " + value_dtype_->ToString() + "]";
}

std::string MapTensorType::ToReprString() const {
  if (IsGeneric()) {
    return "MapTensor";
  }
  MS_EXCEPTION_IF_NULL(key_dtype_);
  MS_EXCEPTION_IF_NULL(value_dtype_);
  return "MapTensor[" + key_dtype_->ToReprString() + ", " + value_dtype_->ToReprString() + "]";
}

std::string MapTensorType::DumpText() const {
  if (IsGeneric()) {
    return "MapTensor";
  }
  MS_EXCEPTION_IF_NULL(key_dtype_);
  MS_EXCEPTION_IF_NULL(value_dtype_);
  return "MapTensor[" + key_dtype_->DumpText() + ", " + value_dtype_->DumpText() + "]";
}

bool MapTensorType::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  const auto &other_type = static_cast<const MapTensorType &>(other);
  return common::IsEqual(key_dtype_, other_type.key_dtype_) && common::IsEqual(value_dtype_, other_type.value_dtype_);
}

size_t MapTensorType::hash() const {
  size_t hash_value = hash_combine(static_cast<size_t>(kMetaTypeObject), static_cast<size_t>(object_type()));
  if (!IsGeneric()) {
    hash_value = hash_combine(hash_value, (key_dtype_ == nullptr ? 0 : key_dtype_->hash()));
    hash_value = hash_combine(hash_value, (value_dtype_ == nullptr) ? 0 : value_dtype_->hash());
  }
  return hash_value;
}
}  // namespace mindspore
