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

#ifndef MINDSPORE_CCSRC_IR_DTYPE_EMPTY_H_
#define MINDSPORE_CCSRC_IR_DTYPE_EMPTY_H_

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
class TypeAnything : public Type {
 public:
  TypeAnything() : Type(kMetaTypeAnything) {}
  ~TypeAnything() override {}
  MS_DECLARE_PARENT(TypeAnything, Type)

  TypeId generic_type_id() const override { return kMetaTypeAnything; }
  TypePtr DeepCopy() const override;
  std::string DumpText() const override { return "AnythingType"; }
};
using TypeAnythingPtr = std::shared_ptr<TypeAnything>;

class TypeNone : public Type {
 public:
  TypeNone() : Type(kMetaTypeNone) {}
  ~TypeNone() override {}
  MS_DECLARE_PARENT(TypeNone, Type)

  TypeId generic_type_id() const override { return kMetaTypeNone; }
  TypePtr DeepCopy() const override { return std::make_shared<TypeNone>(); }
  std::string ToReprString() const override { return "type_none"; }
  std::string DumpText() const override { return "NoneType"; }
};
using TypeNonePtr = std::shared_ptr<TypeNone>;

class TypeNull : public Type {
 public:
  TypeNull() : Type(kMetaTypeNull) {}
  ~TypeNull() override {}
  MS_DECLARE_PARENT(TypeNull, Type)

  TypeId generic_type_id() const override { return kMetaTypeNull; }
  TypePtr DeepCopy() const override { return std::make_shared<TypeNull>(); }
  std::string DumpText() const override { return "NullType"; }
};
using TypeNullPtr = std::shared_ptr<TypeNull>;

class TypeEllipsis : public Type {
 public:
  TypeEllipsis() : Type(kMetaTypeEllipsis) {}
  ~TypeEllipsis() override {}
  MS_DECLARE_PARENT(TypeEllipsis, Type)

  TypeId generic_type_id() const override { return kMetaTypeEllipsis; }
  TypePtr DeepCopy() const override { return std::make_shared<TypeEllipsis>(); }
  std::string ToReprString() const override { return "Ellipsis"; }
  std::string DumpText() const override { return "Ellipsis"; }
};
using TypeEllipsisPtr = std::shared_ptr<TypeEllipsis>;

extern const TypePtr kTypeNone;
extern const TypePtr kTypeNull;
extern const TypePtr kTypeEllipsis;
extern const TypePtr kAnyType;
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_DTYPE_EMPTY_H_
