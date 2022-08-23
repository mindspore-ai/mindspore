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

#ifndef MINDSPORE_CORE_IR_DTYPE_EMPTY_H_
#define MINDSPORE_CORE_IR_DTYPE_EMPTY_H_

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "utils/hash_map.h"
#include "base/base.h"
#include "ir/named.h"
#include "ir/dtype/type.h"

namespace mindspore {
/// \brief TypeAnything defines a Type class whose type is Anything.
class MS_CORE_API TypeAnything : public Type {
 public:
  /// \brief Default constructor for TypeAnything.
  TypeAnything() : Type(kMetaTypeAnything) {}

  /// \brief Destructor of TypeAnything.
  ~TypeAnything() override {}
  MS_DECLARE_PARENT(TypeAnything, Type)

  TypeId generic_type_id() const override { return kMetaTypeAnything; }
  TypePtr DeepCopy() const override;
  std::string DumpText() const override { return "AnythingType"; }
};
using TypeAnythingPtr = std::shared_ptr<TypeAnything>;

/// \brief TypeNone defines a Type class whose type is None.
class MS_CORE_API TypeNone : public Type {
 public:
  /// \brief Default constructor for TypeNone.
  TypeNone() : Type(kMetaTypeNone) {}

  /// \brief Destructor of TypeNone.
  ~TypeNone() override {}
  MS_DECLARE_PARENT(TypeNone, Type)

  TypeId generic_type_id() const override { return kMetaTypeNone; }
  TypePtr DeepCopy() const override { return std::make_shared<TypeNone>(); }
  std::string ToReprString() const override { return "type_none"; }
  std::string DumpText() const override { return "NoneType"; }
};
using TypeNonePtr = std::shared_ptr<TypeNone>;

/// \brief TypeNull defines a Type class whose type is Null.
class MS_CORE_API TypeNull : public Type {
 public:
  /// \brief Default constructor for TypeNull.
  TypeNull() : Type(kMetaTypeNull) {}

  /// \brief Destructor of TypeNull.
  ~TypeNull() override {}
  MS_DECLARE_PARENT(TypeNull, Type)

  TypeId generic_type_id() const override { return kMetaTypeNull; }
  TypePtr DeepCopy() const override { return std::make_shared<TypeNull>(); }
  std::string DumpText() const override { return "NullType"; }
};
using TypeNullPtr = std::shared_ptr<TypeNull>;

/// \brief TypeEllipsis defines a Type class whose type is Ellipsis.
class MS_CORE_API TypeEllipsis : public Type {
 public:
  /// \brief Default constructor for TypeEllipsis.
  TypeEllipsis() : Type(kMetaTypeEllipsis) {}

  /// \brief Destructor of TypeEllipsis.
  ~TypeEllipsis() override {}
  MS_DECLARE_PARENT(TypeEllipsis, Type)

  TypeId generic_type_id() const override { return kMetaTypeEllipsis; }
  TypePtr DeepCopy() const override { return std::make_shared<TypeEllipsis>(); }
  std::string ToReprString() const override { return "Ellipsis"; }
  std::string DumpText() const override { return "Ellipsis"; }
};
using TypeEllipsisPtr = std::shared_ptr<TypeEllipsis>;

GVAR_DEF(TypePtr, kTypeNone, std::make_shared<TypeNone>());
GVAR_DEF(TypePtr, kTypeNull, std::make_shared<TypeNull>());
GVAR_DEF(TypePtr, kTypeEllipsis, std::make_shared<TypeEllipsis>());
GVAR_DEF(TypePtr, kAnyType, std::make_shared<TypeAnything>());
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_EMPTY_H_
