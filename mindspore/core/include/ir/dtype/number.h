/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "utils/log_adapter.h"
#include "base/base.h"
#include "ir/dtype/type.h"

namespace mindspore {
inline TypePtr ToRealType(TypeId type_id, std::string type_name);
inline TypePtr ToComplexType(TypeId type_id, std::string type_name);
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

  size_t ItemSize() const {
    constexpr int kBitsPerByte = 8;
    return nbits_ < kBitsPerByte ? 1 : nbits_ / kBitsPerByte;
  }

  bool IsSigned() const {
    static const std::vector<int> unsigned_types = {kObjectTypeNumber, kNumberTypeBool,   kNumberTypeUInt,
                                                    kNumberTypeGLUInt, kNumberTypeUInt8,  kNumberTypeUInt16,
                                                    kNumberTypeUInt32, kNumberTypeUInt64, kNumberTypeFloat8E8M0FNU};
    return std::find(unsigned_types.begin(), unsigned_types.end(), number_type_) == unsigned_types.end();
  }

  bool IsFloatingPoint() const {
    static const std::vector<int> floating_point_types = {
      kNumberTypeFloat,   kNumberTypeFloat8E4M3FN, kNumberTypeFloat8E5M2, kNumberTypeHiFloat8, kNumberTypeFloat8E8M0FNU,
      kNumberTypeFloat16, kNumberTypeBFloat16,     kNumberTypeFloat32,    kNumberTypeFloat64,  kNumberTypeDouble};
    return std::find(floating_point_types.begin(), floating_point_types.end(), number_type_) !=
           floating_point_types.end();
  }

  bool IsComplex() const {
    static const std::vector<int> complex_types = {kNumberTypeComplex, kNumberTypeComplex64, kNumberTypeComplex128};
    return std::find(complex_types.begin(), complex_types.end(), number_type_) != complex_types.end();
  }

  TypePtr ToReal() const { return ToRealType(number_type_, ToReprString()); }
  TypePtr ToComplex() const { return ToComplexType(number_type_, ToReprString()); }

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
  std::string ToReprString() const override { return "bool"; }
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
  std::string ToReprString() const override { return GetTypeName("int"); }
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

  /// \brief Constructor for Float, used to support float8.
  ///
  /// \param type_id Define the type id of Float object.
  /// \param nbits Define the bit length of Float object.
  explicit Float(const TypeId type_id, const int nbits);

  /// \brief Destructor of Float.
  ~Float() override {}
  MS_DECLARE_PARENT(Float, Number)

  TypeId generic_type_id() const override { return kNumberTypeFloat; }
  TypePtr DeepCopy() const override {
    if (nbits() == 0) {
      return std::make_shared<Float>();
    }
    return std::make_shared<Float>(type_id(), nbits());
  }

  std::string ToString() const override {
    if (type_id() == kNumberTypeFloat8E4M3FN) {
      return "Float8E4M3FN";
    }
    if (type_id() == kNumberTypeFloat8E5M2) {
      return "Float8E5M2";
    }
    if (type_id() == kNumberTypeHiFloat8) {
      return "HiFloat8";
    }
    if (type_id() == kNumberTypeFloat8E8M0FNU) {
      return "Float8E8M0FNU";
    }
    return GetTypeName("Float");
  }

  std::string ToReprString() const override {
    if (type_id() == kNumberTypeFloat8E4M3FN) {
      return "float8_e4m3fn";
    }
    if (type_id() == kNumberTypeFloat8E5M2) {
      return "float8_e5m2";
    }
    if (type_id() == kNumberTypeHiFloat8) {
      return "hifloat8";
    }
    if (type_id() == kNumberTypeFloat8E8M0FNU) {
      return "float8_e8m0fnu";
    }
    return GetTypeName("float");
  }
  std::string DumpText() const override {
    if (type_id() == kNumberTypeFloat8E4M3FN) {
      return "F8E4M3FN";
    }
    if (type_id() == kNumberTypeFloat8E5M2) {
      return "F8E5M2";
    }
    if (type_id() == kNumberTypeHiFloat8) {
      return "HiF8";
    }
    if (type_id() == kNumberTypeFloat8E8M0FNU) {
      return "F8E8M0FNU";
    }
    return nbits() == 0 ? std::string("Float") : std::string("F") + std::to_string(nbits());
  }
};

// BFloat
/// \brief BFloat defines a Number class whose type is brain float.
class MS_CORE_API BFloat : public Number {
 public:
  /// \brief Default constructor for BFloat.
  BFloat() : Number(kNumberTypeBFloat16, 0) {}

  /// \brief Constructor for BFloat.
  ///
  /// \param nbits Define the bit length of BFloat object.
  explicit BFloat(const int nbits);

  /// \brief Destructor of BFloat.
  ~BFloat() override {}
  MS_DECLARE_PARENT(BFloat, Number)

  TypeId generic_type_id() const override { return kNumberTypeBFloat16; }
  TypePtr DeepCopy() const override {
    if (nbits() == 0) {
      return std::make_shared<BFloat>();
    }
    return std::make_shared<BFloat>(nbits());
  }

  std::string ToString() const override { return GetTypeName("BFloat"); }
  std::string ToReprString() const override { return GetTypeName("bfloat"); }
  std::string DumpText() const override {
    return nbits() == 0 ? std::string("BFloat") : std::string("BF") + std::to_string(nbits());
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
GVAR_DEF(TypePtr, kInt4, std::make_shared<Int>(static_cast<int>(BitsNum::eBits4)));
GVAR_DEF(TypePtr, kInt8, std::make_shared<Int>(static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kInt16, std::make_shared<Int>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kInt32, std::make_shared<Int>(static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kInt64, std::make_shared<Int>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kUInt8, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kUInt16, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kUInt32, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kUInt64, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kFloat16, std::make_shared<Float>(kNumberTypeFloat16, static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kFloat32, std::make_shared<Float>(kNumberTypeFloat32, static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kFloat64, std::make_shared<Float>(kNumberTypeFloat64, static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kFloat8E4M3FN, std::make_shared<Float>(kNumberTypeFloat8E4M3FN, static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kFloat8E5M2, std::make_shared<Float>(kNumberTypeFloat8E5M2, static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kHiFloat8, std::make_shared<Float>(kNumberTypeHiFloat8, static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kFloat8E8M0FNU, std::make_shared<Float>(kNumberTypeFloat8E8M0FNU, static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kBFloat16, std::make_shared<BFloat>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kInt, std::make_shared<Int>());
GVAR_DEF(TypePtr, kUInt, std::make_shared<UInt>());
GVAR_DEF(TypePtr, kFloat, std::make_shared<Float>());
GVAR_DEF(TypePtr, kBFloat, std::make_shared<BFloat>());
GVAR_DEF(TypePtr, kNumber, std::make_shared<Number>());
GVAR_DEF(TypePtr, kComplex64, std::make_shared<Complex>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kComplex128, std::make_shared<Complex>(static_cast<int>(BitsNum::eBits128)));

inline TypePtr ToRealType(TypeId type_id, std::string type_name) {
  static const std::unordered_map<TypeId, TypePtr> type_map = {{kNumberTypeComplex64, kFloat32},
                                                               {kNumberTypeComplex128, kFloat64},
                                                               {kNumberTypeBool, kBool},
                                                               {kNumberTypeInt4, kInt4},
                                                               {kNumberTypeInt8, kInt8},
                                                               {kNumberTypeInt16, kInt16},
                                                               {kNumberTypeInt32, kInt32},
                                                               {kNumberTypeInt64, kInt64},
                                                               {kNumberTypeUInt8, kUInt8},
                                                               {kNumberTypeUInt16, kUInt16},
                                                               {kNumberTypeUInt32, kUInt32},
                                                               {kNumberTypeUInt64, kUInt64},
                                                               {kNumberTypeFloat8E4M3FN, kFloat8E4M3FN},
                                                               {kNumberTypeFloat8E5M2, kFloat8E5M2},
                                                               {kNumberTypeHiFloat8, kHiFloat8},
                                                               {kNumberTypeFloat8E8M0FNU, kFloat8E8M0FNU},
                                                               {kNumberTypeFloat16, kFloat16},
                                                               {kNumberTypeFloat32, kFloat32},
                                                               {kNumberTypeFloat64, kFloat64},
                                                               {kNumberTypeBFloat16, kBFloat16}};
  auto it = type_map.find(type_id);
  if (it == type_map.end()) {
    MS_EXCEPTION(TypeError) << "Cannot convert type " << type_name << " to real type.";
  }
  return it->second;
}

inline TypePtr ToComplexType(TypeId type_id, std::string type_name) {
  static const std::unordered_map<TypeId, TypePtr> type_map = {{kNumberTypeComplex64, kComplex64},
                                                               {kNumberTypeBFloat16, kComplex64},
                                                               {kNumberTypeFloat32, kComplex64},
                                                               {kNumberTypeComplex128, kComplex128},
                                                               {kNumberTypeFloat64, kComplex128}};
  auto it = type_map.find(type_id);
  if (it == type_map.end()) {
    MS_EXCEPTION(TypeError) << "Cannot convert type " << type_name << " to complex type.";
  }
  return it->second;
}
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_NUMBER_H_
