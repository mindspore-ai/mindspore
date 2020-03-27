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

#ifndef MINDSPORE_CCSRC_IR_DTYPE_REF_H_
#define MINDSPORE_CCSRC_IR_DTYPE_REF_H_

#include <cstddef>
#include <iostream>
#include <initializer_list>
#include <map>
#include <memory>
#include <utility>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <algorithm>
#include "ir/base.h"
#include "ir/named.h"
#include "ir/dtype/type.h"

namespace mindspore {
// TypeRefKey type
class RefKeyType : public Object {
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
class RefType : public Object {
 public:
  RefType() : Object(kObjectTypeRef) {}
  RefType(const TypePtr& subtype, const TypePtr& subtype_origin)
      : Object(kObjectTypeRef, false), subtype_(subtype), subtype_origin_(subtype_origin) {}
  ~RefType() override {}
  MS_DECLARE_PARENT(RefType, Object)

  TypePtr subtype() const { return subtype_; }
  TypeId generic_type_id() const override { return kObjectTypeRef; }
  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string DumpText() const override;

 private:
  TypePtr subtype_;
  TypePtr subtype_origin_;
};
using RefTypePtr = std::shared_ptr<RefType>;

extern const TypePtr kRefKeyType;
extern const TypePtr kRefType;
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_DTYPE_REF_H_
