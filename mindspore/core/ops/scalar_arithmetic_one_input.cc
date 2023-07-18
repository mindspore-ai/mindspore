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

#include <cmath>
#include <memory>
#include "abstract/ops/op_infer.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "ops/op_utils.h"
#include "ops/scalar_log.h"
#include "ops/scalar_uadd.h"
#include "ops/scalar_usub.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
template <typename T>
ValuePtr UaddImpl(const ValuePtr &x_value) {
  MS_EXCEPTION_IF_NULL(x_value);
  auto x = GetValue<T>(x_value);
  return MakeValue(x);
}

template <typename T>
ValuePtr UsubImpl(const ValuePtr &x_value) {
  MS_EXCEPTION_IF_NULL(x_value);
  auto x = GetValue<T>(x_value);
  return MakeValue(-x);
}

template <typename T>
ValuePtr LogImpl(const ValuePtr &x_value) {
  MS_EXCEPTION_IF_NULL(x_value);
  auto x = GetValue<T>(x_value);
  return MakeValue(static_cast<float>(log(x)));
}

using ScalarImplFunc = std::function<ValuePtr(const ValuePtr &)>;

template <typename T>
ScalarImplFunc ChooseFunction(const std::string &prim_name) {
  std::map<std::string, ScalarImplFunc> infer_value_func_map = {{mindspore::kScalarUaddOpName, UaddImpl<T>},
                                                                {mindspore::kScalarUsubOpName, UsubImpl<T>},
                                                                {mindspore::kScalarLogOpName, LogImpl<T>}};
  auto iter = infer_value_func_map.find(prim_name);
  if (iter == infer_value_func_map.end()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "' don't support. Only support [Uadd, Usub, Log]";
  }
  return iter->second;
}

class ScalarOneInputInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    const int64_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, op_name);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got x: " << elem->ToString();
    }
    return abstract::kNoShape;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto x_type = input_args[0]->BuildType();
    std::set<TypePtr> check_types = {kInt32, kInt64, kFloat32, kFloat64};
    (void)CheckAndConvertUtils::CheckSubClass("x_dtype", x_type, check_types, prim_name);
    if (prim_name == mindspore::kScalarLogOpName) {
      return kFloat32;
    }
    return x_type;
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    const int64_t input_num = 1;
    auto op_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got x: " << elem->ToString();
    }

    auto x_value = elem->BuildValue();
    if (x_value == kValueAny) {
      return nullptr;
    }
    auto x_type = input_args[0]->BuildType();
    ValuePtr result;
    switch (x_type->type_id()) {
      case kNumberTypeInt32: {
        auto func = ChooseFunction<int32_t>(op_name);
        result = func(x_value);
        break;
      }
      case kNumberTypeInt64: {
        auto func = ChooseFunction<int64_t>(op_name);
        result = func(x_value);
        break;
      }
      case kNumberTypeFloat32: {
        auto func = ChooseFunction<float>(op_name);
        result = func(x_value);
        break;
      }
      case kNumberTypeFloat64: {
        auto func = ChooseFunction<double>(op_name);
        result = func(x_value);
        break;
      }
      default: {
        MS_EXCEPTION(TypeError) << "For '" << op_name
                                << "', the supported type is in the list: [int32, int64, float32, float64], but got "
                                << x_type->ToString() << ".";
      }
    }
    return result;
  }
};
MIND_API_OPERATOR_IMPL(ScalarUadd, BaseOperator);
MIND_API_OPERATOR_IMPL(ScalarUsub, BaseOperator);
MIND_API_OPERATOR_IMPL(ScalarLog, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarUadd, prim::kPrimScalarUadd, ScalarOneInputInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarUsub, prim::kPrimScalarUsub, ScalarOneInputInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarLog, prim::kPrimScalarLog, ScalarOneInputInfer, true);
}  // namespace ops
}  // namespace mindspore
