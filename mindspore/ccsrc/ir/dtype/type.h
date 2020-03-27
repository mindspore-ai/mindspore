/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_IR_DTYPE_TYPE_H_
#define MINDSPORE_CCSRC_IR_DTYPE_TYPE_H_

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

namespace mindspore {
//
// Supported meta type
//
enum TypeId : int {
  kTypeUnknown = 0,
  kMetaTypeBegin = kTypeUnknown,
  kMetaTypeType,  // Type
  kMetaTypeAnything,
  kMetaTypeObject,
  kMetaTypeTypeType,  // TypeType
  kMetaTypeProblem,
  kMetaTypeExternal,
  kMetaTypeNone,
  kMetaTypeNull,
  kMetaTypeEnd,
  //
  // Object types
  //
  kObjectTypeBegin = kMetaTypeEnd,
  kObjectTypeNumber,
  kObjectTypeString,
  kObjectTypeList,
  kObjectTypeTuple,
  kObjectTypeSlice,
  kObjectTypeKeyword,
  kObjectTypeTensorType,
  kObjectTypeClass,
  kObjectTypeDictionary,
  kObjectTypeFunction,
  kObjectTypeJTagged,
  kObjectTypeSymbolicKeyType,
  kObjectTypeEnvType,
  kObjectTypeRefKey,
  kObjectTypeRef,
  kObjectTypeEnd,
  //
  // Number Types
  //
  kNumberTypeBegin = kObjectTypeEnd,
  kNumberTypeBool,
  kNumberTypeInt,
  kNumberTypeInt8,
  kNumberTypeInt16,
  kNumberTypeInt32,
  kNumberTypeInt64,
  kNumberTypeUInt,
  kNumberTypeUInt8,
  kNumberTypeUInt16,
  kNumberTypeUInt32,
  kNumberTypeUInt64,
  kNumberTypeFloat,
  kNumberTypeFloat16,
  kNumberTypeFloat32,
  kNumberTypeFloat64,
  kNumberTypeEnd
};

TypeId IntBitsToTypeId(const int nbits);
TypeId UIntBitsToTypeId(const int nbits);
TypeId FloatBitsToTypeId(const int nbits);
const char* TypeIdLabel(const TypeId& v);
TypeId NormalizeTypeId(const TypeId type_id);
bool IsSameObjectType(const Type& lhs, const Type& rhs);
size_t GetTypeByte(const TypePtr& type_ptr);

// Base class for all types
// forward declaration.

class Type : public Value {
 public:
  Type() : meta_type_(kMetaTypeType), is_generic_(true) {}
  explicit Type(TypeId t, bool is_generic = true) : meta_type_(t), is_generic_(is_generic) {}
  ~Type() override = default;
  MS_DECLARE_PARENT(Type, Value)

  bool operator==(const Value& other) const override;
  TypeId meta_type() const { return meta_type_; }

  virtual TypeId type_id() const { return meta_type_; }
  virtual TypeId generic_type_id() const { return kMetaTypeType; }

  virtual bool operator!=(const Type& other) const { return !(*this == other); }
  virtual bool operator==(const Type& other) const { return this->type_id() == other.type_id(); }
  virtual bool equal(const TypePtr other) const { return *this == *other; }

  virtual TypeId object_type() const { return kTypeUnknown; }
  virtual TypeId number_type() const { return kTypeUnknown; }
  virtual TypePtr DeepCopy() const = 0;
  virtual TypePtr Clone() const { return DeepCopy(); }

  std::size_t hash() const override { return std::hash<int>{}(static_cast<int>(type_id())); }

  std::string ToString() const override { return TypeIdLabel(meta_type_); }
  virtual std::string ToReprString() const { return ToString(); }
  std::string ReprString() const { return "mindspore." + ToReprString(); }
  void dump() const override { std::cout << ToString() << std::endl; }
  bool IsUnknown() const { return (meta_type_ == kMetaTypeType); }
  bool IsGeneric() const { return is_generic_; }
  abstract::AbstractBasePtr ToAbstract() override;
  friend std::ostream& operator<<(std::ostream& os, const Type& type);
  friend std::ostream& operator<<(std::ostream& os, const TypePtr type);

  const bool parse_info_ = true;

 private:
  TypeId meta_type_;
  bool is_generic_;
};

using TypePtrList = std::vector<TypePtr>;

//
// Base class for normal objects
//
class Object : public Type {
 public:
  Object() : Type(kMetaTypeObject), object_type_(kMetaTypeObject) {}
  explicit Object(const TypeId object_type, bool is_generic = true)
      : Type(kMetaTypeObject, is_generic), object_type_(object_type) {}
  ~Object() override = default;
  MS_DECLARE_PARENT(Object, Type)

  TypeId object_type() const override { return object_type_; }
  TypeId type_id() const override { return object_type_; }
  TypeId generic_type_id() const override { return kMetaTypeObject; }
  bool equal(const TypePtr other) const override;
  std::string ToString() const override { return std::string("Object:") + TypeIdLabel(object_type_); }

  friend std::ostream& operator<<(std::ostream& os, const Object& obj);
  friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Object> obj);

 private:
  const TypeId object_type_;
};

std::ostream& operator<<(std::ostream& os, const TypePtrList& types);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_DTYPE_TYPE_H_
