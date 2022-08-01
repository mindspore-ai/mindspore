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

#ifndef MINDSPORE_CORE_IR_DTYPE_REF_H_
#define MINDSPORE_CORE_IR_DTYPE_REF_H_

#include <memory>
#include <string>

#include "base/base.h"
#include "ir/named.h"
#include "ir/dtype/type.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {
// TypeRefKey type
/// \brief RefKeyType defines an Object class whose type is RefKey.
class MS_CORE_API RefKeyType final : public Object {
 public:
  /// \brief Default constructor for RefKeyType.
  RefKeyType() : Object(kObjectTypeRefKey) {}

  /// \brief Destructor of RefKeyType.
  ~RefKeyType() override {}
  MS_DECLARE_PARENT(RefKeyType, Object)

  TypeId generic_type_id() const override { return kObjectTypeRefKey; }
  TypePtr DeepCopy() const override { return std::make_shared<RefKeyType>(); }
  std::string ToReprString() const override { return "type_refkey"; }
  std::string DumpText() const override { return "RefKeyType"; }
};

// TypeRef type
/// \brief RefType defines a TensorType class whose type is Ref.
class MS_CORE_API RefType final : public TensorType {
 public:
  /// \brief Default constructor for RefType.
  RefType() : TensorType() {}

  /// \brief Constructor for RefType.
  ///
  /// \param[in] subtype Define the TensorType for RefType object to refer to.
  explicit RefType(const TensorTypePtr &subtype) : TensorType(subtype->element()) {}

  /// \brief Constructor for RefType.
  ///
  /// \param[in] subtype Define the TensorType for RefType object to refer to.
  explicit RefType(const TensorType *subtype) : TensorType(subtype->element()) {}

  /// \brief Destructor of RefType.
  ~RefType() override {}
  MS_DECLARE_PARENT(RefType, TensorType)

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string DumpText() const override;
};
using RefTypePtr = std::shared_ptr<RefType>;

GVAR_DEF(TypePtr, kRefKeyType, std::make_shared<RefKeyType>());
GVAR_DEF(TypePtr, kRefType, std::make_shared<RefType>());
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_REF_H_
