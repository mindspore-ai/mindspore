/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/scalar_arithmetic.h"
#include <set>
#include <limits>
#include <map>
#include <string>
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ScalarArithmeticFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  return abstract::kNoShape;
}

TypePtr ScalarArithmeticFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x_type = input_args[kIndex0]->GetType();
  auto y_type = input_args[kIndex1]->GetType();
  std::set<std::string> compare_ops = {kNameScalarEq, kNameScalarGe, kNameScalarGt, kNameScalarLt, kNameScalarLe};
  auto iter = compare_ops.find(prim_name);
  if (iter != compare_ops.end()) {
    return kBool;
  }
  return HighPriorityType(x_type, y_type, prim_name);
}

template <typename T>
ValuePtr AddImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
#ifndef _MSC_VER
  if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
    T res;
    if (__builtin_add_overflow(x, y, &res)) {
      MS_EXCEPTION(ValueError) << "For prim '" << op_name
                               << "' Overflow of the sum of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return MakeValue(res);
  }
#endif
  return MakeValue(x + y);
}

template <typename T>
ValuePtr SubImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
#ifndef _MSC_VER
  if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
    T res;
    if (__builtin_sub_overflow(x, y, &res)) {
      MS_EXCEPTION(ValueError) << "For prim '" << op_name
                               << "' Overflow of the sub of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return MakeValue(res);
  }
#endif
  return MakeValue(x - y);
}

template <typename T>
ValuePtr MulImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
#ifndef _MSC_VER
  if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
    T res;
    if (__builtin_mul_overflow(x, y, &res)) {
      MS_EXCEPTION(ValueError) << "For prim '" << op_name
                               << "' Overflow of the mul of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
    return MakeValue(res);
  }
#endif
  return MakeValue(x * y);
}

template <typename T>
ValuePtr DivImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
  T zero = 0;
  if (y == zero) {
    MS_EXCEPTION(ValueError) << "The divisor could not be zero. But the divisor is zero now.";
  }
  if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
    if (x == std::numeric_limits<T>::min() && static_cast<int64_t>(y) == -1) {
      MS_EXCEPTION(ValueError) << "For prim '" << op_name
                               << "' Overflow of the div of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
  }
  return MakeValue(static_cast<float>(x) / static_cast<float>(y));
}

template <typename T>
ValuePtr FloorDivImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
  T zero = 0;
  if (y == zero) {
    MS_EXCEPTION(ValueError) << "The divisor could not be zero. But the divisor is zero now.";
  }
  if constexpr (std::is_signed<T>::value) {
    if (x == std::numeric_limits<T>::min() && static_cast<int64_t>(y) == -1) {
      MS_EXCEPTION(ValueError) << "For prim '" << op_name
                               << "' Overflow of the mod of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
  }
  T n = std::floor(static_cast<float>(x) / static_cast<float>(y));
  T mod = x - n * y;
  T res = (x - mod) / y;
  return MakeValue(res);
}

template <typename T>
ValuePtr ModImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
  T zero = 0;
  if (y == zero) {
    MS_EXCEPTION(ValueError) << "Cannot perform modulo operation on zero.";
  }
  if constexpr (std::is_signed<T>::value) {
    if (x == std::numeric_limits<T>::min() && static_cast<int64_t>(y) == -1) {
      MS_EXCEPTION(ValueError) << "For prim '" << op_name
                               << "' Overflow of the mod of two signed number x: " << std::to_string(x)
                               << ", y: " << std::to_string(y) << ".";
    }
  }
  T n = std::floor(static_cast<float>(x) / static_cast<float>(y));
  T res = x - n * y;
  return MakeValue(res);
}

template <typename T>
ValuePtr EqImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x_tmp = GetScalarCastValue<T>(op_name, x_value);
  auto y_tmp = GetScalarCastValue<T>(op_name, y_value);
  auto x = static_cast<double>(x_tmp);
  auto y = static_cast<double>(y_tmp);
  if (std::isinf(x) && std::isinf(y)) {
    return MakeValue((x > 0 && y > 0) || (x < 0 && y < 0));
  }
  double error = x - y;
  error = fabs(error);
  return MakeValue(error < DBL_EPSILON);
}

template <typename T>
ValuePtr LtImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
  return MakeValue(x < y);
}

template <typename T>
ValuePtr GtImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
  return MakeValue(x > y);
}

template <typename T>
ValuePtr LeImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
  return MakeValue(x <= y);
}

template <typename T>
ValuePtr GeImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
  return MakeValue(x >= y);
}

template <typename T>
ValuePtr PowImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarCastValue<T>(op_name, x_value);
  auto y = GetScalarCastValue<T>(op_name, y_value);
  return MakeValue(static_cast<T>(std::pow(x, y)));
}

using MathImplFunc = std::function<ValuePtr(const ValuePtr &, const ValuePtr &, const std::string &)>;

template <typename T>
MathImplFunc ChooseFunc(const std::string &prim_name) {
  std::map<std::string, MathImplFunc> infer_value_func_map = {
    {kNameScalarAdd, AddImpl<T>}, {kNameScalarSub, SubImpl<T>}, {kNameScalarMul, MulImpl<T>},
    {kNameScalarDiv, DivImpl<T>}, {kNameScalarMod, ModImpl<T>}, {kNameScalarEq, EqImpl<T>},
    {kNameScalarGt, GtImpl<T>},   {kNameScalarLt, LtImpl<T>},   {kNameScalarGe, GeImpl<T>},
    {kNameScalarLe, LeImpl<T>},   {kNameScalarPow, PowImpl<T>}, {kNameScalarFloorDiv, FloorDivImpl<T>}};
  auto iter = infer_value_func_map.find(prim_name);
  if (iter == infer_value_func_map.end()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "' don't support. Only support [Add, Sub, Mul, Div, Mod, Eq, Le, Ge, Lt, Gt]";
  }
  return iter->second;
}

ValuePtr ScalarArithmeticFrontendFuncImpl::InferValue(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  constexpr size_t x_index = 0;
  constexpr size_t y_index = 1;
  auto elem_x = input_args[x_index];
  auto elem_y = input_args[y_index];
  if (!CheckAndConvertUtils::IsScalar(elem_x) && !CheckAndConvertUtils::IsScalar(elem_y)) {
    MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got x: " << elem_x->ToString()
                            << " and y: " << elem_y->ToString();
  }

  auto x_value = elem_x->GetValue();
  auto y_value = elem_y->GetValue();
  if (x_value->ContainsValueAny() || y_value->ContainsValueAny()) {
    return nullptr;
  }
  auto x_type = input_args[x_index]->GetType();
  auto y_type = input_args[y_index]->GetType();
  auto res_type = HighPriorityType(x_type, y_type, op_name);
  ValuePtr result;
  switch (res_type->type_id()) {
    case kNumberTypeInt32: {
      auto func = ChooseFunc<int32_t>(op_name);
      result = func(x_value, y_value, op_name);
      break;
    }
    case kNumberTypeInt64: {
      auto func = ChooseFunc<int64_t>(op_name);
      result = func(x_value, y_value, op_name);
      break;
    }
    case kNumberTypeFloat32: {
      auto func = ChooseFunc<float>(op_name);
      result = func(x_value, y_value, op_name);
      break;
    }
    case kNumberTypeFloat64: {
      auto func = ChooseFunc<double>(op_name);
      result = func(x_value, y_value, op_name);
      break;
    }
    case kNumberTypeBool: {
      auto func = ChooseFunc<bool>(op_name);
      result = func(x_value, y_value, op_name);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "For '" << op_name
                              << "', the supported type is in the list: [int32, int64, float32, float64], but got "
                              << res_type->ToString() << ".";
    }
  }
  return result;
}
}  // namespace ops
}  // namespace mindspore
