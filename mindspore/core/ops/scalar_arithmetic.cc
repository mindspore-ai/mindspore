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

#include <vector>
#include <string>
#include <memory>

#include "ops/op_utils.h"
#include "abstract/ops/op_infer.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"
#include "ops/scalar_add.h"
#include "ops/scalar_sub.h"
#include "ops/scalar_mul.h"
#include "ops/scalar_div.h"
#include "ops/scalar_floordiv.h"
#include "ops/scalar_mod.h"
#include "ops/scalar_eq.h"
#include "ops/scalar_lt.h"
#include "ops/scalar_gt.h"
#include "ops/scalar_le.h"
#include "ops/scalar_ge.h"

namespace mindspore {
namespace ops {
template <typename T>
ValuePtr AddImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
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
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
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
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
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
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
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
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
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
  T res = std::floor(x / y);
  return MakeValue(res);
}

template <typename T>
ValuePtr ModImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
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
  auto x_tmp = GetScalarValue<T>(op_name, x_value);
  auto y_tmp = GetScalarValue<T>(op_name, y_value);
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
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
  return MakeValue(x < y);
}

template <typename T>
ValuePtr GtImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
  return MakeValue(x > y);
}

template <typename T>
ValuePtr LeImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
  return MakeValue(x <= y);
}

template <typename T>
ValuePtr GeImpl(const ValuePtr &x_value, const ValuePtr &y_value, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(x_value);
  MS_EXCEPTION_IF_NULL(y_value);
  auto x = GetScalarValue<T>(op_name, x_value);
  auto y = GetScalarValue<T>(op_name, y_value);
  return MakeValue(x >= y);
}

using MathImplFunc = std::function<ValuePtr(const ValuePtr &, const ValuePtr &, const std::string &)>;

template <typename T>
MathImplFunc ChooseFunc(const std::string &prim_name) {
  std::map<std::string, MathImplFunc> infer_value_func_map = {{prim::kScalarAdd, AddImpl<T>},
                                                              {prim::kScalarSub, SubImpl<T>},
                                                              {prim::kScalarMul, MulImpl<T>},
                                                              {prim::kScalarDiv, DivImpl<T>},
                                                              {prim::kScalarMod, ModImpl<T>},
                                                              {prim::kScalarEq, EqImpl<T>},
                                                              {prim::kScalarGt, GtImpl<T>},
                                                              {prim::kScalarLt, LtImpl<T>},
                                                              {prim::kScalarGe, GeImpl<T>},
                                                              {prim::kScalarLe, LeImpl<T>},
                                                              {prim::kScalarFloordiv, FloorDivImpl<T>}};
  auto iter = infer_value_func_map.find(prim_name);
  if (iter == infer_value_func_map.end()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "' don't support. Only support [Add, Sub, Mul, Div, Mod, Eq, Le, Ge, Lt, Gt]";
  }
  return iter->second;
}

class ScalarArithmeticInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr size_t input_len = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, op_name);
    auto elem_x = input_args[0];
    auto elem_y = input_args[kIndex1];
    if (!elem_x->isa<abstract::AbstractScalar>() && !elem_y->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got x: " << elem_x->ToString()
                              << " and y: " << elem_y->ToString();
    }
    return abstract::kNoShape;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto x_type = input_args[0]->BuildType();
    auto y_type = input_args[kIndex1]->BuildType();
    std::set<TypePtr> check_types = {kInt32, kInt64, kFloat32, kFloat64, kBool};
    std::set<std::string> compare_ops = {prim::kScalarEq, prim::kScalarGe, prim::kScalarGt, prim::kScalarLt,
                                         prim::kScalarLe};
    (void)CheckAndConvertUtils::CheckSubClass("x_dtype", x_type, check_types, prim_name);
    (void)CheckAndConvertUtils::CheckSubClass("y_dtype", y_type, check_types, prim_name);
    auto iter = compare_ops.find(prim_name);
    if (prim_name == prim::kScalarDiv) {
      return kFloat32;
    }
    if (iter != compare_ops.end()) {
      return kBool;
    }
    return HighPriorityType(x_type, y_type, prim_name);
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    constexpr size_t input_num = 2;
    auto op_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    constexpr size_t x_index = 0;
    constexpr size_t y_index = 1;
    auto elem_x = input_args[x_index];
    auto elem_y = input_args[y_index];
    if (!elem_x->isa<abstract::AbstractScalar>() && !elem_y->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got x: " << elem_x->ToString()
                              << " and y: " << elem_y->ToString();
    }

    auto x_value = elem_x->BuildValue();
    auto y_value = elem_y->BuildValue();
    if (x_value == kAnyValue || y_value == kAnyValue) {
      return nullptr;
    }
    auto x_type = input_args[x_index]->BuildType();
    auto y_type = input_args[y_index]->BuildType();
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

 private:
  std::function<ValuePtr(const ValuePtr &, const ValuePtr &, const std::string &)> infer_value_func_;
};
MIND_API_OPERATOR_IMPL(ScalarAdd, BaseOperator);
MIND_API_OPERATOR_IMPL(ScalarSub, BaseOperator);
MIND_API_OPERATOR_IMPL(ScalarMul, BaseOperator);
MIND_API_OPERATOR_IMPL(ScalarDiv, BaseOperator);
MIND_API_OPERATOR_IMPL(ScalarFloordiv, BaseOperator);
MIND_API_OPERATOR_IMPL(ScalarMod, BaseOperator);
MIND_API_OPERATOR_IMPL(scalar_eq, BaseOperator);
MIND_API_OPERATOR_IMPL(scalar_gt, BaseOperator);
MIND_API_OPERATOR_IMPL(scalar_ge, BaseOperator);
MIND_API_OPERATOR_IMPL(scalar_lt, BaseOperator);
MIND_API_OPERATOR_IMPL(scalar_le, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarAdd, prim::kPrimScalarAdd, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarSub, prim::kPrimScalarSub, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarMul, prim::kPrimScalarMul, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarDiv, prim::kPrimScalarDiv, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarFloordiv, prim::kPrimScalarFloorDiv, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarMod, prim::kPrimScalarMod, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(scalar_eq, prim::kPrimScalarEq, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(scalar_gt, prim::kPrimScalarGt, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(scalar_ge, prim::kPrimScalarGe, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(scalar_lt, prim::kPrimScalarLt, ScalarArithmeticInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(scalar_le, prim::kPrimScalarLe, ScalarArithmeticInfer, true);
}  // namespace ops
}  // namespace mindspore
