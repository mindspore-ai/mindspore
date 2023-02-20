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

#include "frontend/operator/cc_implementations.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <string>
#include <cfloat>
#include <memory>
#include <type_traits>

#include "utils/log_adapter.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
// namespace to support primitive operators definition
namespace prim {
enum class DataType { kInt, kInt64, kFloat, kDouble, kUnknown };

// Whether has a T type data in AnyPtrList.
template <class T>
bool HasType(const AnyPtrList &list) {
  bool ret = std::any_of(list.begin(), list.end(), [](const AnyPtr &ptr) { return ptr->is<T>(); });
  return ret;
}

DataType InferType(const AnyPtrList &list) {
  if (HasType<double>(list)) {
    return DataType::kDouble;
  } else if (HasType<float>(list)) {
    return DataType::kFloat;
  } else if (HasType<int64_t>(list)) {
    return DataType::kInt64;
  } else if (HasType<int>(list)) {
    return DataType::kInt;
  }
  return DataType::kUnknown;
}

template <typename T>
T InnerScalarAdd(T x, T y) {
#ifndef _MSC_VER
  if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
    T res;
    if (__builtin_add_overflow(x, y, &res)) {
      MS_EXCEPTION(ValueError) << "Overflow of the sum of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return res;
  }
#endif
  return x + y;
}

template <typename T>
T InnerScalarSub(T x, T y) {
#ifndef _MSC_VER
  if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
    T res;
    if (__builtin_sub_overflow(x, y, &res)) {
      MS_EXCEPTION(ValueError) << "Overflow of the sub of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return res;
  }
#endif
  return x - y;
}

template <typename T>
T InnerScalarMul(T x, T y) {
#ifndef _MSC_VER
  if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
    T res;
    if (__builtin_mul_overflow(x, y, &res)) {
      MS_EXCEPTION(ValueError) << "Overflow of the mul of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return res;
  }
#endif
  return x * y;
}

template <typename T>
float InnerScalarDiv(T x, T y) {
  if (y == 0) {
    MS_EXCEPTION(ValueError) << "The divisor could not be zero. But the divisor is zero now.";
  }
  if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
    if (x == std::numeric_limits<T>::min() && static_cast<int64_t>(y) == -1) {
      MS_EXCEPTION(ValueError) << "Overflow of the div of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
  }
  return static_cast<float>(x) / static_cast<float>(y);
}

template <typename T>
T InnerScalarFloordiv(T x, T y) {
  auto ret = std::floor(InnerScalarDiv(x, y));
  return static_cast<T>(ret);
}

template <typename T>
T InnerScalarMod(T x, T y) {
  if (y == 0) {
    MS_EXCEPTION(ValueError) << "Cannot perform modulo operation on zero.";
  }
  if constexpr (!std::is_integral<T>::value) {
    return x - y * std::floor(x / y);
  }
  if constexpr (std::is_signed<T>::value) {
    if (x == std::numeric_limits<T>::min() && static_cast<int64_t>(y) == -1) {
      MS_EXCEPTION(ValueError) << "Overflow of the mod of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
  }
  return static_cast<int64_t>(x) % static_cast<int64_t>(y);
}

template <typename T, typename U>
T InnerScalarPow(T x, U y) {
  return std::pow(x, y);
}

template <typename T, typename U>
bool InnerScalarEq(T x, U y) {
  if (std::isinf(static_cast<double>(x)) && std::isinf(static_cast<double>(y))) {
    return (x > 0 && y > 0) || (x < 0 && y < 0);
  }
  double error = static_cast<double>(x) - static_cast<double>(y);
  error = fabs(error);
  return error < DBL_EPSILON;
}

template <typename T, typename U>
bool InnerScalarLt(T x, U y) {
  return x < y;
}

template <typename T, typename U>
bool InnerScalarGt(T x, U y) {
  return x > y;
}

template <typename T, typename U>
bool InnerScalarNe(T x, U y) {
  return !InnerScalarEq(x, y);
}

template <typename T, typename U>
bool InnerScalarLe(T x, U y) {
  return x <= y;
}

template <typename T, typename U>
bool InnerScalarGe(T x, U y) {
  return x >= y;
}

#define SCALAR_OP(op_t)                                                                                         \
  ValuePtr Scalar##op_t(const ValuePtrList &list) {                                                             \
    constexpr size_t scalar_input_size = 2;                                                                     \
    if (list.size() != scalar_input_size) {                                                                     \
      MS_EXCEPTION(NotSupportError) << "Input number of Scalar" << #op_t << " should be " << scalar_input_size  \
                                    << ", but got " << list.size();                                             \
    }                                                                                                           \
    const ValuePtr &x = list[0];                                                                                \
    const ValuePtr &y = list[1];                                                                                \
    MS_EXCEPTION_IF_NULL(x);                                                                                    \
    MS_EXCEPTION_IF_NULL(y);                                                                                    \
    if (x->isa<FP32Imm>() && y->isa<FP32Imm>()) {                                                               \
      float sum = InnerScalar##op_t(GetValue<float>(x), GetValue<float>(y));                                    \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int32Imm>() && y->isa<Int32Imm>()) {                                                             \
      int sum = InnerScalar##op_t(GetValue<int>(x), GetValue<int>(y));                                          \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int32Imm>() && y->isa<FP32Imm>()) {                                                              \
      float sum = InnerScalar##op_t(IntToFloat(GetValue<int>(x)), GetValue<float>(y));                          \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<FP32Imm>() && y->isa<Int32Imm>()) {                                                              \
      float sum = InnerScalar##op_t(GetValue<float>(x), IntToFloat(GetValue<int>(y)));                          \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int64Imm>() && y->isa<Int64Imm>()) {                                                             \
      int64_t sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<int64_t>(y));                              \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int64Imm>() && y->isa<FP32Imm>()) {                                                              \
      float sum = InnerScalar##op_t(LongToFloat(GetValue<int64_t>(x)), GetValue<float>(y));                     \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int64Imm>() && y->isa<Int32Imm>()) {                                                             \
      int64_t sum = InnerScalar##op_t(GetValue<int64_t>(x), IntToLong(GetValue<int>(y)));                       \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<FP32Imm>() && y->isa<Int64Imm>()) {                                                              \
      float sum = InnerScalar##op_t(GetValue<float>(x), LongToFloat(GetValue<int64_t>(y)));                     \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int32Imm>() && y->isa<Int64Imm>()) {                                                             \
      int64_t sum = InnerScalar##op_t(IntToLong(GetValue<int>(x)), GetValue<int64_t>(y));                       \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<BoolImm>() && y->isa<BoolImm>()) {                                                               \
      int sum = InnerScalar##op_t(static_cast<int>(GetValue<bool>(x)), static_cast<int>(GetValue<bool>(y)));    \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    MS_EXCEPTION(TypeError) << "Unsupported input type for Scalar" << #op_t << ", type of x:" << x->type_name() \
                            << ", value of x:" << x->ToString() << ", type of y:" << y->type_name()             \
                            << ", value of y:" << y->ToString();                                                \
  }

SCALAR_OP(Add)
SCALAR_OP(Sub)
SCALAR_OP(Mul)
SCALAR_OP(Div)
SCALAR_OP(Mod)
SCALAR_OP(Pow)
SCALAR_OP(Floordiv)

#define LOGIC_OP(op_t)                                                                                          \
  ValuePtr Scalar##op_t(const ValuePtrList &list) {                                                             \
    constexpr size_t scalar_input_size = 2;                                                                     \
    if (list.size() != scalar_input_size) {                                                                     \
      MS_EXCEPTION(NotSupportError) << "Input number of Scalar" << #op_t << " should be " << scalar_input_size  \
                                    << ", but got " << list.size();                                             \
    }                                                                                                           \
    const ValuePtr &x = list[0];                                                                                \
    const ValuePtr &y = list[1];                                                                                \
    MS_EXCEPTION_IF_NULL(x);                                                                                    \
    MS_EXCEPTION_IF_NULL(y);                                                                                    \
    if (x->isa<FP32Imm>() && y->isa<FP32Imm>()) {                                                               \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<float>(y));                                     \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int32Imm>() && y->isa<Int32Imm>()) {                                                             \
      bool sum = InnerScalar##op_t(GetValue<int>(x), GetValue<int>(y));                                         \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<FP32Imm>() && y->isa<Int32Imm>()) {                                                              \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<int>(y));                                       \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<FP32Imm>() && y->isa<Int64Imm>()) {                                                              \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<int64_t>(y));                                   \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int32Imm>() && y->isa<FP32Imm>()) {                                                              \
      bool sum = InnerScalar##op_t(GetValue<int>(x), GetValue<float>(y));                                       \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int64Imm>() && y->isa<FP32Imm>()) {                                                              \
      bool sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<float>(y));                                   \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int64Imm>() && y->isa<Int64Imm>()) {                                                             \
      bool sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<int64_t>(y));                                 \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int64Imm>() && y->isa<Int32Imm>()) {                                                             \
      bool sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<int>(y));                                     \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    if (x->isa<Int32Imm>() && y->isa<Int64Imm>()) {                                                             \
      bool sum = InnerScalar##op_t(GetValue<int>(x), GetValue<int64_t>(y));                                     \
      return MakeValue(sum);                                                                                    \
    }                                                                                                           \
    MS_EXCEPTION(TypeError) << "Unsupported input type for Scalar" << #op_t << ", type of x:" << x->type_name() \
                            << ", value of x:" << x->ToString() << ", type of y:" << y->type_name()             \
                            << ", value of y:" << y->ToString();                                                \
  }

LOGIC_OP(Eq)
LOGIC_OP(Lt)
LOGIC_OP(Gt)
LOGIC_OP(Ne)
LOGIC_OP(Le)
LOGIC_OP(Ge)

template <typename T>
T InnerBitAnd(T x, T y) {
  return x & y;
}

template <typename T>
T InnerBitOr(T x, T y) {
  return x | y;
}

template <typename T>
T InnerBitXor(T x, T y) {
  return x ^ y;
}

template <typename T>
T InnerBitLeftShift(T x, T y) {
  if (y < 0) {
    MS_EXCEPTION(ValueError) << "For shift operator, shift count must be a non-negative integer.";
  }
#ifndef _MSC_VER
  if (x == 0) {
    return x;
  }
  if (x < 0) {
    if (x == -1) {
      constexpr T max_bit_count = 64;
      if (y == max_bit_count - 1) {
        return std::numeric_limits<T>::min();
      }
    }
    if (x == std::numeric_limits<T>::min() || static_cast<T>(__builtin_clzll(static_cast<uint64_t>(-x))) <= y) {
      MS_EXCEPTION(RuntimeError) << "Arithmetic left shift causes int64 integer overflow.";
    }
  } else if (static_cast<T>(__builtin_clzll(static_cast<uint64_t>(x))) <= y) {
    MS_EXCEPTION(RuntimeError) << "Arithmetic left shift causes int64 integer overflow.";
  }
#endif
  return x << y;
}

template <typename T>
T InnerBitRightShift(T x, T y) {
  if (y < 0) {
    MS_EXCEPTION(ValueError) << "For shift operator, shift count must be a non-negative integer.";
  }
  return x >> y;
}

#define BIT_OP(op_t)                                                                                      \
  ValuePtr Bit##op_t(const ValuePtrList &list) {                                                          \
    constexpr size_t bit_input_size = 2;                                                                  \
    if (list.size() != bit_input_size) {                                                                  \
      MS_EXCEPTION(NotSupportError) << "Input number of Bit" << #op_t << " should be" << bit_input_size   \
                                    << ", but got " << list.size();                                       \
    }                                                                                                     \
    const ValuePtr &x = list[0];                                                                          \
    const ValuePtr &y = list[1];                                                                          \
    MS_EXCEPTION_IF_NULL(x);                                                                              \
    MS_EXCEPTION_IF_NULL(y);                                                                              \
    if (x->isa<Int32Imm>() && y->isa<Int32Imm>()) {                                                       \
      int32_t res = InnerBit##op_t(IntToLong(GetValue<int>(x)), IntToLong(GetValue<int>(y)));             \
      return MakeValue(res);                                                                              \
    }                                                                                                     \
    if (x->isa<Int64Imm>() && y->isa<Int32Imm>()) {                                                       \
      int64_t res = InnerBit##op_t(GetValue<int64_t>(x), IntToLong(GetValue<int>(y)));                    \
      return MakeValue(res);                                                                              \
    }                                                                                                     \
    if (x->isa<Int32Imm>() && y->isa<Int64Imm>()) {                                                       \
      int64_t res = InnerBit##op_t(IntToLong(GetValue<int>(x)), GetValue<int64_t>(y));                    \
      return MakeValue(res);                                                                              \
    }                                                                                                     \
    if (x->isa<Int64Imm>() && y->isa<Int64Imm>()) {                                                       \
      int64_t res = InnerBit##op_t(GetValue<int64_t>(x), GetValue<int64_t>(y));                           \
      return MakeValue(res);                                                                              \
    }                                                                                                     \
    MS_EXCEPTION(TypeError) << "Unsupported input type. For Bit" << #op_t                                 \
                            << ", only integer types are supported, but got type of x:" << x->type_name() \
                            << ", value of x:" << x->ToString() << ", type of y:" << y->type_name()       \
                            << ", value of y:" << y->ToString();                                          \
  }

BIT_OP(And)
BIT_OP(Or)
BIT_OP(Xor)
BIT_OP(LeftShift)
BIT_OP(RightShift)

ValuePtr ScalarUAdd(const ValuePtrList &list) {
  constexpr size_t scalar_input_size = 1;
  if (list.size() != scalar_input_size) {
    MS_EXCEPTION(NotSupportError) << "Input number of ScalarUAdd should be " << scalar_input_size << ", but got "
                                  << list.size() << ".";
  }
  const auto &x = list[0];
  MS_EXCEPTION_IF_NULL(x);
  return x;
}

ValuePtr ScalarUSub(const ValuePtrList &list) {
  constexpr size_t scalar_input_size = 1;
  if (list.size() != scalar_input_size) {
    MS_EXCEPTION(NotSupportError) << "Input number of ScalarUSub should be " << scalar_input_size << ", but got "
                                  << list.size() << ".";
  }
  const auto &x = list[0];
  MS_EXCEPTION_IF_NULL(x);

  if (x->isa<Int32Imm>()) {
    int32_t sum = -1 * GetValue<int32_t>(x);
    return MakeValue(sum);
  }
  if (x->isa<Int64Imm>()) {
    int64_t sum = -1 * GetValue<int64_t>(x);
    return MakeValue(sum);
  }
  if (x->isa<FP32Imm>()) {
    float sum = -1.0f * GetValue<float>(x);
    return MakeValue(sum);
  }
  MS_EXCEPTION(NotSupportError) << "Not support ScalarUSub [x:" << x->ToString() << "].";
}

ValuePtr ScalarLog(const ValuePtrList &list) {
  constexpr size_t scalar_input_size = 1;
  if (list.size() != scalar_input_size) {
    MS_EXCEPTION(NotSupportError) << "Input number of ScalarLog must be " << scalar_input_size << ", but got "
                                  << list.size() << ".";
  }
  const auto &x = list[0];
  MS_EXCEPTION_IF_NULL(x);

  if (x->isa<FP32Imm>()) {
    auto v = static_cast<float>(log(GetValue<float>(x)));
    return MakeValue(v);
  }
  MS_EXCEPTION(NotSupportError) << "Not support ScalarLog [x:" << x->ToString() << "].";
}

void GetBooleansFromValueList(const std::string &prim_name, const ValuePtrList &list, bool *val_x, bool *val_y) {
  constexpr size_t boolean_input_size = 2;
  if (list.size() != boolean_input_size) {
    MS_EXCEPTION(NotSupportError) << "The input number of " << prim_name << " operator must be " << boolean_input_size
                                  << ", but got " << list.size() << ".";
  }
  const auto &x = list[0];
  const auto &y = list[1];
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  if (!x->isa<BoolImm>() || !y->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << "The inputs of " << prim_name
                      << " operator should be two booleans, but got param0: " << x->ToString()
                      << ", param1: " << y->ToString() << ".";
  }
  *val_x = x->cast<BoolImmPtr>()->value();
  *val_y = y->cast<BoolImmPtr>()->value();
}

void GetStringsFromValueList(const std::string &prim_name, const ValuePtrList &list, std::string *str_x,
                             std::string *str_y) {
  constexpr size_t string_input_size = 2;
  if (list.size() != string_input_size) {
    MS_EXCEPTION(NotSupportError) << "The input number of " << prim_name << " operator must be " << string_input_size
                                  << ", but got " << list.size() << ".";
  }
  const auto &x = list[0];
  const auto &y = list[1];
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  if (!x->isa<StringImm>() || !y->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The inputs of " << prim_name
                      << " operator should be two strings, but got param0: " << x->ToString()
                      << ", param1: " << y->ToString() << ".";
  }
  *str_x = GetValue<std::string>(x);
  *str_y = GetValue<std::string>(y);
}

ValuePtr BoolNot(const ValuePtrList &list) {
  constexpr size_t boolean_input_size = 1;
  if (list.size() != boolean_input_size) {
    MS_EXCEPTION(NotSupportError) << "Input number of BoolNot must be " << boolean_input_size << ", but got "
                                  << list.size() << ".";
  }
  const auto &x = list[0];
  MS_EXCEPTION_IF_NULL(x);
  if (!x->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << "The input of BoolNot operator should be a boolean, but got " << x->ToString() << ".";
  }
  bool val = x->cast<BoolImmPtr>()->value();
  return MakeValue<bool>(!val);
}

ValuePtr StringNot(const ValuePtrList &list) {
  constexpr size_t string_input_size = 1;
  if (list.size() != string_input_size) {
    MS_EXCEPTION(NotSupportError) << "Input number of StringNot must be " << string_input_size << ", but got "
                                  << list.size() << ".";
  }
  const auto &x = list[0];
  MS_EXCEPTION_IF_NULL(x);
  if (!x->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The input of BoolNot operator should be a string, but got " << x->ToString() << ".";
  }
  std::string str = x->cast<StringImmPtr>()->value();
  return MakeValue<bool>(str.empty());
}

ValuePtr BoolAnd(const ValuePtrList &list) {
  bool x = false;
  bool y = false;
  GetBooleansFromValueList("BoolAnd", list, &x, &y);
  return MakeValue<bool>(x && y);
}

ValuePtr BoolOr(const ValuePtrList &list) {
  bool x = false;
  bool y = false;
  GetBooleansFromValueList("BoolOr", list, &x, &y);
  return MakeValue<bool>(x || y);
}

ValuePtr BoolEq(const ValuePtrList &list) {
  bool x = false;
  bool y = false;
  GetBooleansFromValueList("BoolEq", list, &x, &y);
  return MakeValue<bool>(x == y);
}

ValuePtr StringEq(const ValuePtrList &list) {
  std::string str_x;
  std::string str_y;
  GetStringsFromValueList("StringEq", list, &str_x, &str_y);
  return MakeValue<bool>(str_x == str_y);
}

ValuePtr StringLt(const ValuePtrList &list) {
  std::string str_x;
  std::string str_y;
  GetStringsFromValueList("StringLt", list, &str_x, &str_y);
  return MakeValue<bool>(str_x < str_y);
}

ValuePtr StringGt(const ValuePtrList &list) {
  std::string str_x;
  std::string str_y;
  GetStringsFromValueList("StringGt", list, &str_x, &str_y);
  return MakeValue<bool>(str_x > str_y);
}

ValuePtr StringLe(const ValuePtrList &list) {
  std::string str_x;
  std::string str_y;
  GetStringsFromValueList("StringLe", list, &str_x, &str_y);
  return MakeValue<bool>(str_x <= str_y);
}

ValuePtr StringGe(const ValuePtrList &list) {
  std::string str_x;
  std::string str_y;
  GetStringsFromValueList("StringGe", list, &str_x, &str_y);
  return MakeValue<bool>(str_x >= str_y);
}

ValuePtr StringIn(const ValuePtrList &list) {
  std::string str_x;
  std::string str_y;
  GetStringsFromValueList("StringIn", list, &str_x, &str_y);
  return MakeValue<bool>(str_y.find(str_x) != std::string::npos);
}

ValuePtr StringConcat(const ValuePtrList &list) {
  std::string str_x;
  std::string str_y;
  GetStringsFromValueList("StringConcat", list, &str_x, &str_y);
  return MakeValue<std::string>(str_x + str_y);
}
}  // namespace prim
}  // namespace mindspore
