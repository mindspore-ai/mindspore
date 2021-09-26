/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_DTYPE_TYPE_H_
#define MINDSPORE_CORE_IR_DTYPE_TYPE_H_

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

#include "base/base.h"
#include "ir/named.h"
#include "ir/dtype/type_id.h"

namespace mindspore {

TypeId IntBitsToTypeId(const int nbits);
TypeId UIntBitsToTypeId(const int nbits);
TypeId FloatBitsToTypeId(const int nbits);
TypeId ComplexBitsToTypeId(const int nbits);
MS_CORE_API const std::string &TypeIdLabel(const TypeId &v);
TypeId NormalizeTypeId(const TypeId type_id);
bool IsSameObjectType(const Type &lhs, const Type &rhs);
size_t GetTypeByte(const TypePtr &type_ptr);

enum class BitsNum : int {
  eBits8 = 8,
  eBits16 = 16,
  eBits32 = 32,
  eBits64 = 64,
  eBits128 = 128,
};

// Base class for all types
// forward declaration.
class MS_CORE_API Type : public Value {
 public:
  Type() : meta_type_(kMetaTypeType), is_generic_(true) {}
  explicit Type(TypeId t, bool is_generic = true) : meta_type_(t), is_generic_(is_generic) {}
  ~Type() override = default;
  MS_DECLARE_PARENT(Type, Value)

  bool operator==(const Value &other) const override;
  TypeId meta_type() const { return meta_type_; }

  virtual TypeId type_id() const { return meta_type_; }
  virtual TypeId generic_type_id() const { return kMetaTypeType; }

  virtual bool operator!=(const Type &other) const { return !(*this == other); }
  virtual bool operator==(const Type &other) const { return this->type_id() == other.type_id(); }
  virtual bool equal(const TypePtr other) const { return *this == *other; }

  virtual TypeId object_type() const { return kTypeUnknown; }
  virtual TypeId parent_type() const { return kTypeUnknown; }
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
  friend std::ostream &operator<<(std::ostream &os, const Type &type);
  friend std::ostream &operator<<(std::ostream &os, const TypePtr type);

 private:
  TypeId meta_type_;
  bool is_generic_;
};

using TypePtrList = std::vector<TypePtr>;

//
// Base class for normal objects
//
class MS_CORE_API Object : public Type {
 public:
  Object() : Type(kMetaTypeObject), object_type_(kMetaTypeObject), parent_type_(kMetaTypeObject) {}
  explicit Object(const TypeId object_type, bool is_generic = true)
      : Type(kMetaTypeObject, is_generic), object_type_(object_type), parent_type_(kMetaTypeObject) {}
  explicit Object(const TypeId object_type, const TypeId parent_type, bool is_generic = true)
      : Type(kMetaTypeObject, is_generic), object_type_(object_type), parent_type_(parent_type) {}
  ~Object() override = default;
  MS_DECLARE_PARENT(Object, Type)

  TypeId object_type() const override { return object_type_; }
  TypeId parent_type() const override { return parent_type_; }
  TypeId type_id() const override { return object_type_; }
  TypeId generic_type_id() const override { return kMetaTypeObject; }
  bool equal(const TypePtr other) const override;
  std::string ToString() const override { return std::string("Object:") + TypeIdLabel(object_type_); }

  friend std::ostream &operator<<(std::ostream &os, const Object &obj);
  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Object> obj);

 private:
  const TypeId object_type_;
  const TypeId parent_type_;
};

//
// TypeId name map
//
const std::unordered_map<TypeId, std::string> type_name_map = {
  {kNumberTypeBool, "bool_"},      {kNumberTypeInt8, "int8"},       {kNumberTypeUInt8, "uint8"},
  {kNumberTypeInt16, "int16"},     {kNumberTypeInt32, "int32"},     {kNumberTypeInt64, "int64"},
  {kNumberTypeFloat16, "float16"}, {kNumberTypeFloat32, "float32"}, {kNumberTypeFloat64, "float64"}};

const std::unordered_map<TypeId, int> type_priority_map = {
  {kNumberTypeBool, 0},    {kNumberTypeUInt8, 1},   {kNumberTypeInt8, 2},
  {kNumberTypeInt16, 3},   {kNumberTypeInt32, 4},   {kNumberTypeInt64, 5},
  {kNumberTypeFloat16, 6}, {kNumberTypeFloat32, 7}, {kNumberTypeFloat64, 8}};

MS_CORE_API std::ostream &operator<<(std::ostream &os, const TypePtrList &types);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_TYPE_H_
