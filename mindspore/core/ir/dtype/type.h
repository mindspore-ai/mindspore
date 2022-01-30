/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "utils/hash_map.h"
#include "base/base.h"
#include "ir/named.h"
#include "ir/dtype/type_id.h"

namespace mindspore {

TypeId IntBitsToTypeId(const int nbits);
TypeId UIntBitsToTypeId(const int nbits);
TypeId FloatBitsToTypeId(const int nbits);
TypeId ComplexBitsToTypeId(const int nbits);

/// \brief Get label of the input TypeId.
///
/// \param[in] v Define the input TypeId.
/// \return The label of input TypeId.
MS_CORE_API const std::string &TypeIdLabel(const TypeId &v);
MS_CORE_API TypeId NormalizeTypeId(const TypeId type_id);
bool IsSameObjectType(const Type &lhs, const Type &rhs);
MS_CORE_API size_t GetTypeByte(const TypePtr &type_ptr);

enum class BitsNum : int {
  eBits8 = 8,
  eBits16 = 16,
  eBits32 = 32,
  eBits64 = 64,
  eBits128 = 128,
};

/// \brief Type defines an Value class for type.
class MS_CORE_API Type : public Value {
 public:
  /// \brief Default constructor for Type.
  Type() : meta_type_(kMetaTypeType), is_generic_(true) {}

  /// \brief Constructor for Type.
  ///
  /// \param[in] t Define TypeId for Type object.
  /// \param[in] is_generic Define whether the Type object is generic.
  explicit Type(TypeId t, bool is_generic = true) : meta_type_(t), is_generic_(is_generic) {}

  /// \brief Destructor of Type.
  ~Type() override = default;
  MS_DECLARE_PARENT(Type, Value)

  bool operator==(const Value &other) const override;

  /// \brief Show the meta type of the Type object.
  ///
  /// \return The meta type of the Type object.
  TypeId meta_type() const { return meta_type_; }

  /// \brief Show the type id of the Type object.
  ///
  /// \return The type id of the Type object.
  virtual TypeId type_id() const { return meta_type_; }

  /// \brief Show the generic type id for the Number object.
  ///
  /// \return The generic type id.
  virtual TypeId generic_type_id() const { return kMetaTypeType; }

  /// \brief Check whether the input is not the current Type object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Check whether the current object and other object are different.
  virtual bool operator!=(const Type &other) const { return !(*this == other); }

  /// \brief Check whether the input is the current Type object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Check whether the current object and other object have the same type id.
  virtual bool operator==(const Type &other) const { return this->type_id() == other.type_id(); }

  /// \brief Check whether the input is the current Type object.
  ///
  /// \param[in] other Define a TypePtr.
  /// \return Check whether the current object and other object are the same.
  virtual bool equal(const TypePtr other) const { return *this == *other; }

  /// \brief Get the object type of the Type object.
  ///
  /// \return The object type of the Type object.
  virtual TypeId object_type() const { return kTypeUnknown; }

  /// \brief Get the parent type of the Type object.
  ///
  /// \return The parent type of the Type object.
  virtual TypeId parent_type() const { return kTypeUnknown; }

  /// \brief Get the number type of the Type object.
  ///
  /// \return The number type of the Type object.
  virtual TypeId number_type() const { return kTypeUnknown; }

  /// \brief Deep copy the Type object.
  ///
  /// \return The deep copy of the Type object.
  virtual TypePtr DeepCopy() const = 0;

  /// \brief Clone the Type object.
  ///
  /// \return The clone of the Type object.
  virtual TypePtr Clone() const { return DeepCopy(); }

  std::size_t hash() const override { return std::hash<int>{}(static_cast<int>(type_id())); }
  std::string ToString() const override { return TypeIdLabel(meta_type_); }

  /// \brief Get Type object ToReprString description.
  ///
  /// \return The description of Type object.
  virtual std::string ToReprString() const { return ToString(); }

  /// \brief Get Type object ToReprString description.
  ///
  /// \return The description of Type object.
  std::string ReprString() const { return "mindspore." + ToReprString(); }
  void dump() const override { std::cout << ToString() << std::endl; }

  /// \brief Check whether the Type object is unknown.
  ///
  /// \return whether the Type object is unknown.
  bool IsUnknown() const { return (meta_type_ == kMetaTypeType); }

  /// \brief Check whether the Type object is generic.
  ///
  /// \return whether the Type object is generic.
  bool IsGeneric() const { return is_generic_; }
  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief Get Type object ToString description.
  ///
  /// \param[in] os The ostream to receive the description
  /// \param[in] type The Type object need to show the description
  /// \return The ostream with Type object description
  MS_CORE_API friend std::ostream &operator<<(std::ostream &os, const Type &type);

  /// \brief Get Type object ToString description.
  ///
  /// \param[in] os The ostream to receive the description
  /// \param[in] type The TypePtr need to show the description
  /// \return The ostream with Type object description
  MS_CORE_API friend std::ostream &operator<<(std::ostream &os, const TypePtr type);

 private:
  TypeId meta_type_;
  bool is_generic_;
};

using TypePtrList = std::vector<TypePtr>;

/// \brief Type defines an Type class for object.
class MS_CORE_API Object : public Type {
 public:
  /// \brief Default constructor for Object.
  Object() : Type(kMetaTypeObject), object_type_(kMetaTypeObject), parent_type_(kMetaTypeObject) {}

  /// \brief Constructor for Object.
  ///
  /// \param[in] object_type Define object type for Object object.
  /// \param[in] is_generic Define whether the Object object is generic.
  explicit Object(const TypeId object_type, bool is_generic = true)
      : Type(kMetaTypeObject, is_generic), object_type_(object_type), parent_type_(kMetaTypeObject) {}

  /// \brief Constructor for Object.
  ///
  /// \param[in] object_type Define object type for Object object.
  /// \param[in] parent_type Define the parent type for Object object.
  /// \param[in] is_generic Define whether the Object object is generic.
  explicit Object(const TypeId object_type, const TypeId parent_type, bool is_generic = true)
      : Type(kMetaTypeObject, is_generic), object_type_(object_type), parent_type_(parent_type) {}

  /// \brief Destructor of Object.
  ~Object() override = default;
  MS_DECLARE_PARENT(Object, Type)

  TypeId object_type() const override { return object_type_; }
  TypeId parent_type() const override { return parent_type_; }
  TypeId type_id() const override { return object_type_; }
  TypeId generic_type_id() const override { return kMetaTypeObject; }
  bool equal(const TypePtr other) const override;
  std::string ToString() const override { return std::string("Object:") + TypeIdLabel(object_type_); }

  /// \brief Get Object object ToString description.
  ///
  /// \param[in] os The ostream to receive the description
  /// \param[in] obj The Object object need to show the description
  /// \return The ostream with Object object description
  friend std::ostream &operator<<(std::ostream &os, const Object &obj);

  /// \brief Get Object object ToString description.
  ///
  /// \param[in] os The ostream to receive the description
  /// \param[in] obj The Object object need to show the description
  /// \return The ostream with Object object description
  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Object> obj);

 private:
  const TypeId object_type_;
  const TypeId parent_type_;
};

//
// TypeId name map
//
const mindspore::HashMap<TypeId, std::string> type_name_map = {
  {kNumberTypeBool, "bool_"},      {kNumberTypeInt8, "int8"},       {kNumberTypeUInt8, "uint8"},
  {kNumberTypeInt16, "int16"},     {kNumberTypeInt32, "int32"},     {kNumberTypeInt64, "int64"},
  {kNumberTypeFloat16, "float16"}, {kNumberTypeFloat32, "float32"}, {kNumberTypeFloat64, "float64"}};

const mindspore::HashMap<TypeId, int> type_priority_map = {
  {kNumberTypeBool, 0},    {kNumberTypeUInt8, 1},   {kNumberTypeInt8, 2},
  {kNumberTypeInt16, 3},   {kNumberTypeInt32, 4},   {kNumberTypeInt64, 5},
  {kNumberTypeFloat16, 6}, {kNumberTypeFloat32, 7}, {kNumberTypeFloat64, 8}};

/// \brief Get TypePtrList description.
///
/// \param[in] os The ostream to receive the description
/// \param[in] types The TypePtrList need to show the description
/// \return The ostream with TypePtrList description
MS_CORE_API std::ostream &operator<<(std::ostream &os, const TypePtrList &types);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_TYPE_H_
