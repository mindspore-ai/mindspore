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

#ifndef MINDSPORE_CORE_IR_DTYPE_NUMBER_H_
#define MINDSPORE_CORE_IR_DTYPE_NUMBER_H_

#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "utils/hash_map.h"
#include "base/base.h"
#include "ir/named.h"
#include "ir/dtype/type.h"

namespace mindspore {
/// \brief Number defines an Object class whose type is number.
class MS_CORE_API Number : public Object {
 public:
  /// \brief Default constructor for Number.
  Number() : Object(kObjectTypeNumber), number_type_(kObjectTypeNumber), nbits_(0) {}

  /// \brief Constructor for  Number.
  ///
  /// \param[in] number_type Define the number type of Number object.
  /// \param[in] nbits Define the bit length of Number object.
  /// \param[in] is_generic Define whether it is generic for Number object.
  Number(const TypeId number_type, const int nbits, bool is_generic = true)
      : Object(kObjectTypeNumber, is_generic), number_type_(number_type), nbits_(nbits) {}

  /// \brief Destructor of Number.
  ~Number() override = default;
  MS_DECLARE_PARENT(Number, Object)

  /// \brief Get the bit length of Number object.
  ///
  /// \return bit length of Number object.
  int nbits() const { return nbits_; }

  TypeId number_type() const override { return number_type_; }
  TypeId type_id() const override { return number_type_; }
  TypeId generic_type_id() const override { return kObjectTypeNumber; }
  bool operator==(const Type &other) const override;
  std::size_t hash() const override;
  TypePtr DeepCopy() const override { return std::make_shared<Number>(); }
  std::string ToString() const override { return "Number"; }
  std::string ToReprString() const override { return "number"; }
  std::string DumpText() const override { return "Number"; }

  /// \brief Get type name for Number object.
  ///
  /// \param type_name Define the type name.
  /// \return The full type name of the Number object.
  std::string GetTypeName(const std::string &type_name) const {
    std::ostringstream oss;
    oss << type_name;
    if (nbits() != 0) {
      oss << nbits();
    }
    return oss.str();
  }

 private:
  const TypeId number_type_;
  const int nbits_;
};

using NumberPtr = std::shared_ptr<Number>;

// Bool
/// \brief Bool defines a Number class whose type is boolean.
class MS_CORE_API Bool : public Number {
 public:
  /// \brief Default constructor for Bool.
  Bool() : Number(kNumberTypeBool, 8) {}

  /// \brief Destructor of Bool.
  ~Bool() override = default;
  MS_DECLARE_PARENT(Bool, Number)

  TypeId generic_type_id() const override { return kNumberTypeBool; }
  TypePtr DeepCopy() const override { return std::make_shared<Bool>(); }
  std::string ToString() const override { return "Bool"; }
  std::string ToReprString() const override { return "bool_"; }
  std::string DumpText() const override { return "Bool"; }
};

// Int
/// \brief Int defines a Number class whose type is int.
class MS_CORE_API Int : public Number {
 public:
  /// \brief Default constructor for Int.
  Int() : Number(kNumberTypeInt, 0) {}

  /// \brief Constructor for Int.
  ///
  /// \param nbits Define the bit length of Int object.
  explicit Int(const int nbits);

  /// \brief Destructor of Int.
  ~Int() override = default;
  MS_DECLARE_PARENT(Int, Number)

  TypeId generic_type_id() const override { return kNumberTypeInt; }
  TypePtr DeepCopy() const override {
    if (nbits() == 0) {
      return std::make_shared<Int>();
    }
    return std::make_shared<Int>(nbits());
  }

  std::string ToString() const override { return GetTypeName("Int"); }
  std::string ToReprString() const override { return nbits() == 0 ? "int_" : GetTypeName("int"); }
  std::string DumpText() const override {
    return nbits() == 0 ? std::string("Int") : std::string("I") + std::to_string(nbits());
  }
};

// UInt
/// \brief UInt defines a Number class whose type is uint.
class MS_CORE_API UInt : public Number {
 public:
  /// \brief Default constructor for UInt.
  UInt() : Number(kNumberTypeUInt, 0) {}

  /// \brief Constructor for UInt.
  ///
  /// \param nbits Define the bit length of UInt object.
  explicit UInt(const int nbits);

  TypeId generic_type_id() const override { return kNumberTypeUInt; }

  /// \brief Destructor of UInt.
  ~UInt() override {}
  MS_DECLARE_PARENT(UInt, Number)

  TypePtr DeepCopy() const override {
    if (nbits() == 0) {
      return std::make_shared<UInt>();
    }
    return std::make_shared<UInt>(nbits());
  }

  std::string ToString() const override { return GetTypeName("UInt"); }
  std::string ToReprString() const override { return GetTypeName("uint"); }
  std::string DumpText() const override {
    return nbits() == 0 ? std::string("UInt") : std::string("U") + std::to_string(nbits());
  }
};

// Float
/// \brief Float defines a Number class whose type is float.
class MS_CORE_API Float : public Number {
 public:
  /// \brief Default constructor for Float.
  Float() : Number(kNumberTypeFloat, 0) {}

  /// \brief Constructor for Float.
  ///
  /// \param nbits Define the bit length of Float object.
  explicit Float(const int nbits);

  /// \brief Destructor of Float.
  ~Float() override {}
  MS_DECLARE_PARENT(Float, Number)

  TypeId generic_type_id() const override { return kNumberTypeFloat; }
  TypePtr DeepCopy() const override {
    if (nbits() == 0) {
      return std::make_shared<Float>();
    }
    return std::make_shared<Float>(nbits());
  }

  std::string ToString() const override { return GetTypeName("Float"); }
  std::string ToReprString() const override { return nbits() == 0 ? "float_" : GetTypeName("float"); }
  std::string DumpText() const override {
    return nbits() == 0 ? std::string("Float") : std::string("F") + std::to_string(nbits());
  }
};

// Complex
/// \brief Complex defines a Number class whose type is complex.
class MS_CORE_API Complex : public Number {
 public:
  /// \brief Default constructor for Complex.
  Complex() : Number(kNumberTypeComplex, 0) {}

  /// \brief Constructor for Complex.
  ///
  /// \param nbits Define the bit length of Complex object.
  explicit Complex(const int nbits);

  /// \brief Destructor of Complex.
  ~Complex() override {}
  MS_DECLARE_PARENT(Complex, Number)

  TypeId generic_type_id() const override { return kNumberTypeComplex; }
  TypePtr DeepCopy() const override {
    if (nbits() == 0) {
      return std::make_shared<Complex>();
    }
    return std::make_shared<Complex>(nbits());
  }

  std::string ToString() const override { return GetTypeName("Complex"); }
  std::string ToReprString() const override { return GetTypeName("complex"); }
  std::string DumpText() const override { return std::string("Complex") + std::to_string(nbits()); }
};

GVAR_DEF(TypePtr, kBool, std::make_shared<Bool>());
GVAR_DEF(TypePtr, kInt8, std::make_shared<Int>(static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kInt16, std::make_shared<Int>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kInt32, std::make_shared<Int>(static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kInt64, std::make_shared<Int>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kUInt8, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kUInt16, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kUInt32, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kUInt64, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kFloat16, std::make_shared<Float>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kFloat32, std::make_shared<Float>(static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kFloat64, std::make_shared<Float>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kInt, std::make_shared<Int>());
GVAR_DEF(TypePtr, kUInt, std::make_shared<UInt>());
GVAR_DEF(TypePtr, kFloat, std::make_shared<Float>());
GVAR_DEF(TypePtr, kNumber, std::make_shared<Number>());
GVAR_DEF(TypePtr, kComplex64, std::make_shared<Complex>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kComplex128, std::make_shared<Complex>(static_cast<int>(BitsNum::eBits128)));
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_NUMBER_H_
