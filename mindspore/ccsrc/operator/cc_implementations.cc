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

#include "operator/cc_implementations.h"
#include <cassert>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include "utils/misc.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "common/utils.h"

namespace mindspore {
// namespace to support primitive operators definition
namespace prim {
enum class DataType { kInt, kFloat, kDouble, kUnknown };

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
  } else if (HasType<int>(list)) {
    return DataType::kInt;
  }
  return DataType::kUnknown;
}

enum OpType { ADD, SUB, MUL, DIV, MOD };

template <typename T>
bool IsSignedIntOverflow(T x, T y, OpType opType) {
  auto max = std::numeric_limits<T>::max();
  auto min = std::numeric_limits<T>::min();

  if (opType == OpType::ADD) {
    return (y > 0 && (max - y) < x) || (y < 0 && (min - y) > x);
  }

  if (opType == OpType::SUB) {
    return (y < 0 && (max + y) < x) || (y > 0 && (min + y) > x);
  }

  if (opType == OpType::MUL) {
    return (x > 0 && y > 0 && (max / y) < x) || (x < 0 && y < 0 && (max / y) > x) ||
           (x > 0 && y < 0 && (min / y) < x) || (x < 0 && y > 0 && (min / y) > x);
  }

  if (opType == OpType::DIV || opType == OpType::MOD) {
    return x == min && static_cast<int64_t>(y) == -1;
  }

  MS_LOG(EXCEPTION) << "Unsupported operation type.";
}

template <typename T>
T InnerScalarAdd(T x, T y) {
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::ADD)) {
    MS_LOG(EXCEPTION) << "Overflow of the sum of two signed number x: " << std::to_string(x)
                      << ", y: " << std::to_string(y) << ".";
  }
  return x + y;
}

template <typename T>
T InnerScalarSub(T x, T y) {
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::SUB)) {
    MS_LOG(EXCEPTION) << "Overflow of the sub of two signed number x: " << std::to_string(x)
                      << ", y: " << std::to_string(y) << ".";
  }
  return x - y;
}

template <typename T>
T InnerScalarMul(T x, T y) {
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::MUL)) {
    MS_LOG(EXCEPTION) << "Overflow of the mul of two signed number x: " << std::to_string(x)
                      << ", y: " << std::to_string(y) << ".";
  }
  return x * y;
}

template <typename T>
float InnerScalarDiv(T x, T y) {
  if (y == 0) {
    MS_LOG(EXCEPTION) << "Divisor could not be zero";
  }
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::DIV)) {
    MS_LOG(EXCEPTION) << "Overflow of the div of two signed number x: " << std::to_string(x)
                      << ", y: " << std::to_string(y) << ".";
  }
  return static_cast<float>(x) / static_cast<float>(y);
}

template <typename T>
T InnerScalarFloordiv(T x, T y) {
  auto ret = std::floor(InnerScalarDiv(x, y));
  if (std::is_integral<T>::value) {
    return static_cast<int>(ret);
  }
  return ret;
}

template <typename T>
T InnerScalarMod(T x, T y) {
  if (y == 0) {
    MS_LOG(EXCEPTION) << "Could not mod to zero.";
  }
  if (std::is_integral<T>::value && std::is_signed<T>::value && IsSignedIntOverflow(x, y, OpType::MOD)) {
    MS_LOG(EXCEPTION) << "Overflow of the mod of two signed number x: " << std::to_string(x)
                      << ", y: " << std::to_string(y) << ".";
  }
  if (std::is_integral<T>::value) {
    return static_cast<int>(x) % static_cast<int>(y);
  }
  float x_int = std::floor(x);
  float y_int = std::ceil(y);
  float max = x_int / y_int;
  float ret = x - y * max;
  return ret;
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

#define SCALAR_OP(op_t)                                                                        \
  ValuePtr Scalar##op_t(const ValuePtrList &list) {                                            \
    do {                                                                                       \
      if (list.size() < 2) {                                                                   \
        MS_LOG(EXCEPTION) << "length of input list for Scalar" << #op_t << " is less than 2."; \
      }                                                                                        \
      ValuePtr x = list[0];                                                                    \
      ValuePtr y = list[1];                                                                    \
      MS_EXCEPTION_IF_NULL(x);                                                                 \
      MS_EXCEPTION_IF_NULL(y);                                                                 \
      if (x->isa<FP64Imm>() && y->isa<FP64Imm>()) {                                            \
        double sum = InnerScalar##op_t(GetValue<double>(x), GetValue<double>(y));              \
        return MakeValue(sum);                                                                 \
      }                                                                                        \
      if (x->isa<FP32Imm>() && y->isa<FP32Imm>()) {                                            \
        float sum = InnerScalar##op_t(GetValue<float>(x), GetValue<float>(y));                 \
        return MakeValue(sum);                                                                 \
      }                                                                                        \
      if (x->isa<Int32Imm>() && y->isa<Int32Imm>()) {                                          \
        int sum = InnerScalar##op_t(GetValue<int>(x), GetValue<int>(y));                       \
        return MakeValue(sum);                                                                 \
      }                                                                                        \
      if (x->isa<Int32Imm>() && y->isa<FP32Imm>()) {                                           \
        float sum = InnerScalar##op_t(IntToFloat(GetValue<int>(x)), GetValue<float>(y));       \
        return MakeValue(sum);                                                                 \
      }                                                                                        \
      if (x->isa<FP32Imm>() && y->isa<Int32Imm>()) {                                           \
        float sum = InnerScalar##op_t(GetValue<float>(x), IntToFloat(GetValue<int>(y)));       \
        return MakeValue(sum);                                                                 \
      }                                                                                        \
      MS_LOG(EXCEPTION) << "Unsupported Value for Scalar" << #op_t << ", x: " << x->ToString() \
                        << ", y: " << y->ToString();                                           \
    } while (0);                                                                               \
  }

SCALAR_OP(Add)
SCALAR_OP(Sub)
SCALAR_OP(Mul)
SCALAR_OP(Div)
SCALAR_OP(Mod)
SCALAR_OP(Pow)
SCALAR_OP(Floordiv)

#define LOGIC_OP(op_t)                                                                       \
  ValuePtr Scalar##op_t(const ValuePtrList &list) {                                          \
    if (list.size() < 2) {                                                                   \
      MS_LOG(EXCEPTION) << "length of input list for Scalar" << #op_t << " is less than 2."; \
    }                                                                                        \
    ValuePtr x = list[0];                                                                    \
    ValuePtr y = list[1];                                                                    \
    MS_EXCEPTION_IF_NULL(x);                                                                 \
    MS_EXCEPTION_IF_NULL(y);                                                                 \
    if (x->isa<FP64Imm>() && y->isa<FP64Imm>()) {                                            \
      bool sum = InnerScalar##op_t(GetValue<double>(x), GetValue<double>(y));                \
      return MakeValue(sum);                                                                 \
    }                                                                                        \
    if (x->isa<FP32Imm>() && y->isa<FP32Imm>()) {                                            \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<float>(y));                  \
      return MakeValue(sum);                                                                 \
    }                                                                                        \
    if (x->isa<FP64Imm>() && y->isa<FP32Imm>()) {                                            \
      bool sum = InnerScalar##op_t(GetValue<double>(x), GetValue<float>(y));                 \
      return MakeValue(sum);                                                                 \
    }                                                                                        \
    if (x->isa<FP32Imm>() && y->isa<FP64Imm>()) {                                            \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<double>(y));                 \
      return MakeValue(sum);                                                                 \
    }                                                                                        \
    if (x->isa<Int32Imm>() && y->isa<Int32Imm>()) {                                          \
      bool sum = InnerScalar##op_t(GetValue<int>(x), GetValue<int>(y));                      \
      return MakeValue(sum);                                                                 \
    }                                                                                        \
    if (x->isa<FP32Imm>() && y->isa<Int32Imm>()) {                                           \
      bool sum = InnerScalar##op_t(GetValue<float>(x), GetValue<int>(y));                    \
      return MakeValue(sum);                                                                 \
    }                                                                                        \
    if (x->isa<Int32Imm>() && y->isa<FP32Imm>()) {                                           \
      bool sum = InnerScalar##op_t(GetValue<int>(x), GetValue<float>(y));                    \
      return MakeValue(sum);                                                                 \
    }                                                                                        \
    if (x->isa<Int64Imm>() && y->isa<Int32Imm>()) {                                          \
      bool sum = InnerScalar##op_t(GetValue<int64_t>(x), GetValue<int>(y));                  \
      return MakeValue(sum);                                                                 \
    }                                                                                        \
    MS_LOG(EXCEPTION) << "Unsupported Value for Scalar" << #op_t << ", x: " << x->ToString() \
                      << ", y: " << y->ToString() << ".";                                    \
  }

LOGIC_OP(Eq)
LOGIC_OP(Lt)
LOGIC_OP(Gt)
LOGIC_OP(Ne)
LOGIC_OP(Le)
LOGIC_OP(Ge)

ValuePtr ScalarUAdd(const ValuePtrList &list) {
  if (list.size() != 1) {
    MS_LOG(EXCEPTION) << "Input number of ScalarUAdd should be 1, but got " << list.size();
  }
  ValuePtr x = list[0];
  MS_EXCEPTION_IF_NULL(x);
  return x;
}

ValuePtr ScalarUSub(const ValuePtrList &list) {
  if (list.size() != 1) {
    MS_LOG(EXCEPTION) << "Input number of ScalarUSub should be 1, but got " << list.size();
  }
  ValuePtr x = list[0];
  MS_EXCEPTION_IF_NULL(x);

  if (x->isa<Int32Imm>()) {
    int32_t sum = -1 * GetValue<int>(x);
    return MakeValue(sum);
  }
  if (x->isa<FP32Imm>()) {
    float sum = -1.0f * GetValue<float>(x);
    return MakeValue(sum);
  }

  MS_LOG(EXCEPTION) << "Unsported Value for ScalarUSub, x: " << x->ToString() << ".";
}

ValuePtr ScalarLog(const ValuePtrList &list) {
  if (list.empty()) {
    MS_LOG(EXCEPTION) << "Input list of ScalarLog is empty.";
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

  MS_LOG(EXCEPTION) << "Unsported Value for ScalarLog, x: " << x->ToString();
}

ValuePtr BoolNot(const ValuePtrList &list) {
  if (list.empty()) {
    MS_LOG(EXCEPTION) << "value list of BoolNot is empty";
  }
  ValuePtr x = list[0];
  MS_EXCEPTION_IF_NULL(x);
  bool convert = false;

  if (ValueToBool(x, &convert)) {
    auto res = !convert;
    return MakeValue(res);
  }

  MS_LOG(EXCEPTION) << "Unsported Value for BoolNot, x: " << x->ToString();
}

ValuePtr BoolAnd(const ValuePtrList &list) {
  if (list.size() < 2) {
    MS_LOG(EXCEPTION) << "Input number " << list.size() << " of BoolAnd is less then 2.";
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

  MS_LOG(EXCEPTION) << "Unsported Value for BoolAnd, x: " << x->ToString() << ".";
}

ValuePtr BoolOr(const ValuePtrList &list) {
  if (list.size() < 2) {
    MS_LOG(EXCEPTION) << "Input number " << list.size() << " of BoolOr is less then 2.";
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

  MS_LOG(EXCEPTION) << "Unsported Value for BoolOr, x: " << x->ToString() << ".";
}

ValuePtr BoolEq(const ValuePtrList &list) {
  if (list.size() < 2) {
    MS_LOG(EXCEPTION) << "Input number " << list.size() << " of BoolEq is less than 2.";
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

  MS_LOG(EXCEPTION) << "Unsported Value for BoolEq, x: " << x->ToString() << ".";
}

std::vector<int> BroadcastShape_(std::vector<int> shpx, std::vector<int> shpy) {
  int dlen = SizeToInt(shpx.size()) - SizeToInt(shpy.size());
  if (dlen < 0) {
    for (int i = 0; i < -dlen; ++i) {
      (void)shpx.insert(shpx.begin(), 1);
    }
  } else if (dlen > 0) {
    for (int i = 0; i < dlen; i++) {
      (void)shpy.insert(shpy.begin(), 1);
    }
  }
  if (shpx.size() != shpy.size()) {
    MS_LOG(EXCEPTION) << "Failure: shpx.size() != shpy.size().";
  }
  std::vector<int> shp;
  for (size_t i = 0; i < shpx.size(); i++) {
    auto a = shpx[i];
    auto b = shpy[i];
    if (a == 1) {
      shp.push_back(b);
    } else if (b == 1) {
      shp.push_back(a);
    } else if (a == -1) {
      shp.push_back(b);
    } else if (b == -1) {
      shp.push_back(a);
    } else if (a == b) {
      shp.push_back(a);
    } else {
      return std::vector<int>();
    }
  }
  return shp;
}
}  // namespace prim
}  // namespace mindspore
