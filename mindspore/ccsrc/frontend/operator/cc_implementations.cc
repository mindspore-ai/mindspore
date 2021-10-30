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

#include "frontend/operator/cc_implementations.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "utils/ms_utils.h"

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
bool IsAddOverflow(const T &x, const T &y, const T &max, const T &min) {
  return (y > 0 && (max - y) < x) || (y < 0 && (min - y) > x);
}

template <typename T>
bool IsSubOverflow(const T &x, const T &y, const T &max, const T &min) {
  return (y < 0 && (max + y) < x) || (y > 0 && (min + y) > x);
}

template <typename T>
bool IsMulOverflow(const T &x, const T &y, const T &max, const T &min) {
  return (x > 0 && y > 0 && (max / y) < x) || (x < 0 && y < 0 && (max / y) > x) || (x > 0 && y < 0 && (min / y) < x) ||
         (x < 0 && y > 0 && (min / y) > x);
}

template <typename T>
bool IsDivOverflow(const T &x, const T &y, const T &min) {
  return (x == min && static_cast<int64_t>(y) == -1);
}

enum class OpType { ADD, SUB, MUL, DIV, MOD };

template <typename T>
bool IsSignedIntOverflow(T x, T y, OpType opType) {
  auto max = std::numeric_limits<T>::max();
  auto min = std::numeric_limits<T>::min();

  if (opType == OpType::ADD) {
    return IsAddOverflow<T>(x, y, max, min);
  }

  if (opType == OpType::SUB) {
    return IsSubOverflow<T>(x, y, max, min);
  }

  if (opType == OpType::MUL) {
    return IsMulOverflow<T>(x, y, max, min);
  }

  if (opType == OpType::DIV || opType == OpType::MOD) {
    return IsDivOverflow<T>(x, y, min);
  }

  MS_EXCEPTION(NotSupportError) << "Unsupported operation type.";
}

template <typename T>
T InnerScalarAdd(T x, T y) {
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::ADD)) {
    MS_EXCEPTION(ValueError) << "Overflow of the sum of two signed number x: " << std::to_string(x)
                             << ", y: " << std::to_string(y) << ".";
  }
  return x + y;
}

template <typename T>
T InnerScalarSub(T x, T y) {
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::SUB)) {
    MS_EXCEPTION(ValueError) << "Overflow of the sub of two signed number x: " << std::to_string(x)
                             << ", y: " << std::to_string(y) << ".";
  }
  return x - y;
}

template <typename T>
T InnerScalarMul(T x, T y) {
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::MUL)) {
    MS_EXCEPTION(ValueError) << "Overflow of the mul of two signed number x: " << std::to_string(x)
                             << ", y: " << std::to_string(y) << ".";
  }
  return x * y;
}

template <typename T>
float InnerScalarDiv(T x, T y) {
  if (y == 0) {
    MS_EXCEPTION(ValueError) << "Divisor could not be zero";
  }
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::DIV)) {
    MS_EXCEPTION(ValueError) << "Overflow of the div of two signed number x: " << std::to_string(x)
                             << ", y: " << std::to_string(y) << ".";
  }
  return static_cast<float>(x) / static_cast<float>(y);
}

template <typename T>
T InnerScalarFloordiv(T x, T y) {
  auto ret = std::floor(InnerScalarDiv(x, y));
  if (std::is_integral<T>::value) {
    return static_cast<int64_t>(ret);
  }
  return ret;
}

template <typename T>
T InnerScalarMod(T x, T y) {
  if (y == 0) {
    MS_EXCEPTION(ValueError) << "Could not mod to zero.";
  }
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::MOD)) {
    MS_EXCEPTION(ValueError) << "Overflow of the mod of two signed number x: " << std::to_string(x)
                             << ", y: " << std::to_string(y) << ".";
  }
  if (std::is_integral<T>::value) {
    return static_cast<int64_t>(x) % static_cast<int64_t>(y);
  }
  return x - y * std::floor(x / y);
}

template <typename T, typename U>
T InnerScalarPow(T x, U y) {
  return std::pow(x, y);
}

template <typename T, typename U>
bool InnerScalarEq(T x, U y) {
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

#define SCALAR_OP(op_t)                                                                                                \
  ValuePtr Scalar##op_t(const ValuePtrList &list) {                                                                    \
    do {                                                                                                               \
      if (list.size() != 2) {                                                                                          \
        MS_EXCEPTION(NotSupportError) << "Input number of Scalar" << #op_t << " should be 2, but got " << list.size(); \
      }                                                                                                                \
      ValuePtr x = list[0];                                                                                            \
      ValuePtr y = list[1];                                                                                            \
      MS_EXCEPTION_IF_NULL(x);                                                                                         \
      MS_EXCEPTION_IF_NULL(y);                                                                                         \
      if (x->isa<FP64Imm>() && y->isa<FP64Imm>()) {                                                                    \
        double sum = InnerScalar##op_t(GetValue<double>(x), GetValue<double>(y));                                      \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<FP32Imm>() && y->isa<FP32Imm>()) {                                                                    \
        float sum = InnerScalar##op_t(GetValue<float>(x), GetValue<float>(y));                                         \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<Int32Imm>() && y->isa<Int32Imm>()) {                                                                  \
        int sum = InnerScalar##op_t(GetValue<int>(x), GetValue<int>(y));                                               \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<Int32Imm>() && y->isa<FP32Imm>()) {                                                                   \
        float sum = InnerScalar##op_t(IntToFloat(GetValue<int>(x)), GetValue<float>(y));                               \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<FP32Imm>() && y->isa<Int32Imm>()) {                                                                   \
        float sum = InnerScalar##op_t(GetValue<float>(x), IntToFloat(GetValue<int>(y)));                               \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<Int64Imm>() && y->isa<Int64Imm>()) {                                                                  \
        int64_t sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<int64_t>(y));                                   \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<Int64Imm>() && y->isa<FP64Imm>()) {                                                                   \
        double sum = InnerScalar##op_t(LongToDouble(GetValue<int64_t>(x)), GetValue<double>(y));                       \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<Int64Imm>() && y->isa<FP32Imm>()) {                                                                   \
        double sum = InnerScalar##op_t(LongToDouble(GetValue<int64_t>(x)), FloatToDouble(GetValue<float>(y)));         \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<Int64Imm>() && y->isa<Int32Imm>()) {                                                                  \
        int64_t sum = InnerScalar##op_t(GetValue<int64_t>(x), IntToLong(GetValue<int>(y)));                            \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<FP32Imm>() && y->isa<Int64Imm>()) {                                                                   \
        double sum = InnerScalar##op_t(FloatToDouble(GetValue<float>(x)), LongToDouble(GetValue<int64_t>(y)));         \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<FP64Imm>() && y->isa<Int64Imm>()) {                                                                   \
        double sum = InnerScalar##op_t(GetValue<double>(x), LongToDouble(GetValue<int64_t>(y)));                       \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      if (x->isa<Int32Imm>() && y->isa<Int64Imm>()) {                                                                  \
        int64_t sum = InnerScalar##op_t(IntToLong(GetValue<int>(x)), GetValue<int64_t>(y));                            \
        return MakeValue(sum);                                                                                         \
      }                                                                                                                \
      MS_EXCEPTION(TypeError) << "Unsupported input type for Scalar" << #op_t << ", type of x:" << x->type_name()      \
                              << ", value of x:" << x->ToString() << ", type of y:" << y->type_name()                  \
                              << ", value of y:" << y->ToString();                                                     \
    } while (0);                                                                                                       \
  }

SCALAR_OP(Add)
SCALAR_OP(Sub)
SCALAR_OP(Mul)
SCALAR_OP(Div)
SCALAR_OP(Mod)
SCALAR_OP(Pow)
SCALAR_OP(Floordiv)

#define LOGIC_OP(op_t)                                                                                               \
  ValuePtr Scalar##op_t(const ValuePtrList &list) {                                                                  \
    if (list.size() != 2) {                                                                                          \
      MS_EXCEPTION(NotSupportError) << "Input number of Scalar" << #op_t << " should be 2, but got " << list.size(); \
    }                                                                                                                \
    ValuePtr x = list[0];                                                                                            \
    ValuePtr y = list[1];                                                                                            \
    MS_EXCEPTION_IF_NULL(x);                                                                                         \
    MS_EXCEPTION_IF_NULL(y);                                                                                         \
    if (x->isa<FP64Imm>() && y->isa<FP64Imm>()) {                                                                    \
      bool sum = InnerScalar##op_t(GetValue<double>(x), GetValue<double>(y));                                        \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<FP32Imm>() && y->isa<FP32Imm>()) {                                                                    \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<float>(y));                                          \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<FP64Imm>() && y->isa<FP32Imm>()) {                                                                    \
      bool sum = InnerScalar##op_t(GetValue<double>(x), GetValue<float>(y));                                         \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<FP32Imm>() && y->isa<FP64Imm>()) {                                                                    \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<double>(y));                                         \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<Int32Imm>() && y->isa<Int32Imm>()) {                                                                  \
      bool sum = InnerScalar##op_t(GetValue<int>(x), GetValue<int>(y));                                              \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<FP32Imm>() && y->isa<Int32Imm>()) {                                                                   \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<int>(y));                                            \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<FP32Imm>() && y->isa<Int64Imm>()) {                                                                   \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<int64_t>(y));                                        \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<Int32Imm>() && y->isa<FP32Imm>()) {                                                                   \
      bool sum = InnerScalar##op_t(GetValue<int>(x), GetValue<float>(y));                                            \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<Int64Imm>() && y->isa<FP32Imm>()) {                                                                   \
      bool sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<float>(y));                                        \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<Int64Imm>() && y->isa<Int64Imm>()) {                                                                  \
      bool sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<int64_t>(y));                                      \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<FP64Imm>() && y->isa<Int64Imm>()) {                                                                   \
      bool sum = InnerScalar##op_t(GetValue<double>(x), GetValue<int64_t>(y));                                       \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<Int64Imm>() && y->isa<FP64Imm>()) {                                                                   \
      bool sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<double>(y));                                       \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<Int64Imm>() && y->isa<Int32Imm>()) {                                                                  \
      bool sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<int>(y));                                          \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    if (x->isa<Int32Imm>() && y->isa<Int64Imm>()) {                                                                  \
      bool sum = InnerScalar##op_t(GetValue<int>(x), GetValue<int64_t>(y));                                          \
      return MakeValue(sum);                                                                                         \
    }                                                                                                                \
    MS_EXCEPTION(TypeError) << "Unsupported input type for Scalar" << #op_t << ", type of x:" << x->type_name()      \
                            << ", value of x:" << x->ToString() << ", type of y:" << y->type_name()                  \
                            << ", value of y:" << y->ToString();                                                     \
  }

LOGIC_OP(Eq)
LOGIC_OP(Lt)
LOGIC_OP(Gt)
LOGIC_OP(Ne)
LOGIC_OP(Le)
LOGIC_OP(Ge)

ValuePtr ScalarUAdd(const ValuePtrList &list) {
  if (list.size() != 1) {
    MS_EXCEPTION(NotSupportError) << "Input number of ScalarUAdd should be 1, but got " << list.size();
  }
  ValuePtr x = list[0];
  MS_EXCEPTION_IF_NULL(x);
  return x;
}

ValuePtr ScalarUSub(const ValuePtrList &list) {
  if (list.size() != 1) {
    MS_EXCEPTION(NotSupportError) << "Input number of ScalarUSub should be 1, but got " << list.size();
  }
  ValuePtr x = list[0];
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
  if (list.size() != 1) {
    MS_EXCEPTION(NotSupportError) << "Input number of ScalarLog must be 1, but got " << list.size();
  }
  ValuePtr x = list[0];
  MS_EXCEPTION_IF_NULL(x);

  if (x->isa<FP64Imm>()) {
    double v = log(GetValue<double>(x));
    return MakeValue(v);
  }
  if (x->isa<FP32Imm>()) {
    auto v = static_cast<float>(log(GetValue<float>(x)));
    return MakeValue(v);
  }

  MS_EXCEPTION(NotSupportError) << "Not support ScalarLog [x:" << x->ToString() << "].";
}

ValuePtr BoolNot(const ValuePtrList &list) {
  if (list.size() != 1) {
    MS_EXCEPTION(NotSupportError) << "Input number of BoolNot must be 1, but got " << list.size();
  }
  ValuePtr x = list[0];
  MS_EXCEPTION_IF_NULL(x);
  bool convert = false;

  if (ValueToBool(x, &convert)) {
    auto res = !convert;
    return MakeValue(res);
  }

  MS_EXCEPTION(NotSupportError) << "Not support BoolNot [x:" << x->ToString() << "].";
}

ValuePtr BoolAnd(const ValuePtrList &list) {
  if (list.size() != 2) {
    MS_EXCEPTION(NotSupportError) << "Input number of BoolAnd must be 2, but got " << list.size();
  }
  ValuePtr x = list[0];
  ValuePtr y = list[1];
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  bool x_b = false;
  bool y_b = false;

  if (ValueToBool(x, &x_b) && ValueToBool(y, &y_b)) {
    auto res = x_b && y_b;
    return MakeValue(res);
  }

  MS_EXCEPTION(NotSupportError) << "Not support [x:" << x->ToString() << "] BoolAnd [y:" << y->ToString();
}

ValuePtr BoolOr(const ValuePtrList &list) {
  if (list.size() != 2) {
    MS_EXCEPTION(NotSupportError) << "Input number of BoolOr must be 2, but got " << list.size();
  }
  ValuePtr x = list[0];
  ValuePtr y = list[1];
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  bool x_b = false;
  bool y_b = false;

  if (ValueToBool(x, &x_b) && ValueToBool(y, &y_b)) {
    auto res = x_b || y_b;
    return MakeValue(res);
  }

  MS_EXCEPTION(NotSupportError) << "Not support [x:" << x->ToString() << "] BoolOr [y:" << y->ToString() << "].";
}

ValuePtr BoolEq(const ValuePtrList &list) {
  if (list.size() != 2) {
    MS_EXCEPTION(NotSupportError) << "Input number of BoolEq must be 2, but got " << list.size();
  }
  ValuePtr x = list[0];
  ValuePtr y = list[1];
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  bool x_b = false;
  bool y_b = false;

  if (ValueToBool(x, &x_b) && ValueToBool(y, &y_b)) {
    auto res = x_b == y_b;
    return MakeValue(res);
  }

  MS_EXCEPTION(NotSupportError) << "Not support [x:" << x->ToString() << "] BoolEq [y:" << y->ToString() << "].";
}
}  // namespace prim
}  // namespace mindspore
