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
class MS_CORE_API RefKeyType : public Object {
 public:
  RefKeyType() : Object(kObjectTypeRefKey) {}
  ~RefKeyType() override {}
  MS_DECLARE_PARENT(RefKeyType, Object)

  TypeId generic_type_id() const override { return kObjectTypeRefKey; }
  TypePtr DeepCopy() const override { return std::make_shared<RefKeyType>(); }
  std::string ToReprString() const override { return "type_refkey"; }
  std::string DumpText() const override { return "RefKeyType"; }
};

// TypeRef type
class MS_CORE_API RefType : public TensorType {
 public:
  RefType() : TensorType() {}
  explicit RefType(const TensorTypePtr &subtype) : TensorType(subtype->element()) {}
  ~RefType() override {}
  MS_DECLARE_PARENT(RefType, TensorType)

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string DumpText() const override;
};
using RefTypePtr = std::shared_ptr<RefType>;

inline const TypePtr kRefKeyType = std::make_shared<RefKeyType>();
inline const TypePtr kRefType = std::make_shared<RefType>();
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_REF_H_
