/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <string>
#include <cstdlib>
#include <algorithm>
#include "utils/log_adapter.h"

namespace mindspore {
TypePtr UndeterminedType::DeepCopy() const {
  MS_EXCEPTION_IF_NULL(element_type_);
  if (IsGeneric()) {
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
  auto other_elem_type = static_cast<const UndeterminedType &>(other).element_type_;
  if (element_type_ == nullptr && other_elem_type == nullptr) {
    return true;
  } else if (element_type_ == nullptr || other_elem_type == nullptr) {
    return false;
  }
  return *element_type_ == *other_elem_type;
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
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  auto other_elem_type = static_cast<const TensorType &>(other).element_type_;
  // When element_type_ = nullptr, which means any type of Array.
  if (element_type_ == nullptr && other_elem_type == nullptr) {
    return true;
  } else if (element_type_ == nullptr || other_elem_type == nullptr) {
    return false;
  }
  return *element_type_ == *other_elem_type;
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
  auto other_elem_type = static_cast<const RowTensorType &>(other).element_type_;
  if (element_type_ == nullptr && other_elem_type == nullptr) {
    return true;
  } else if (element_type_ == nullptr || other_elem_type == nullptr) {
    return false;
  }
  return *element_type_ == *other_elem_type;
}

TypePtr SparseTensorType::DeepCopy() const {
  MS_EXCEPTION_IF_NULL(element_type_);
  if (IsGeneric()) {
    return std::make_shared<SparseTensorType>();
  }
  return std::make_shared<SparseTensorType>(element_type_->DeepCopy());
}

std::string SparseTensorType::ToReprString() const {
  if (element_type_ == nullptr) {
    return "SparseTensor";
  }
  return "SparseTensor[" + element_type_->ToReprString() + "]";
}

std::string SparseTensorType::ToString() const {
  if (element_type_ == nullptr) {
    return "SparseTensor";
  }
  return "SparseTensor[" + element_type_->ToString() + "]";
}

std::string SparseTensorType::DumpText() const {
  if (element_type_ == nullptr) {
    return "SparseTensor";
  }
  return "SparseTensor[" + element_type_->DumpText() + "]";
}

bool SparseTensorType::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  auto other_elem_type = static_cast<const SparseTensorType &>(other).element_type_;
  if (element_type_ == nullptr && other_elem_type == nullptr) {
    return true;
  } else if (element_type_ == nullptr || other_elem_type == nullptr) {
    return false;
  }
  return *element_type_ == *other_elem_type;
}
}  // namespace mindspore
