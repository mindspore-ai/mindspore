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

#include "ops/ops_frontend_func_impl/scalar_arithmetic_unary.h"
#include <map>
#include <string>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
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

template <typename T>
ValuePtr BoolImpl(const ValuePtr &x_value) {
  MS_EXCEPTION_IF_NULL(x_value);
  auto x = GetValue<T>(x_value);
  return MakeValue(static_cast<bool>(x));
}

using ScalarImplFunc = std::function<ValuePtr(const ValuePtr &)>;

template <typename T>
ScalarImplFunc ChooseFunction(const std::string &prim_name) {
  std::map<std::string, ScalarImplFunc> infer_value_func_map = {{kNameScalarUadd, UaddImpl<T>},
                                                                {kNameScalarUsub, UsubImpl<T>},
                                                                {kNameScalarLog, LogImpl<T>},
                                                                {kNameScalarBool, BoolImpl<T>}};
  auto iter = infer_value_func_map.find(prim_name);
  if (iter == infer_value_func_map.end()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "' don't support. Only support [Bool, Uadd, Usub, Log]";
  }
  return iter->second;
}
}  // namespace

ValuePtr ScalarArithmeticUnaryFrontendFuncImpl::InferValue(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto elem = input_args[0];
  if (!CheckAndConvertUtils::IsScalar(elem)) {
    MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got x: " << elem->ToString();
  }

  auto x_value = elem->GetValue();
  if (x_value->ContainsValueAny()) {
    return nullptr;
  }
  auto x_type = input_args[0]->GetType();
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
    case kNumberTypeBool: {
      if (op_name == kNameScalarBool) {
        result = MakeValue(static_cast<bool>(x_value));
      } else {
        MS_EXCEPTION(TypeError) << "For '" << op_name << ", [bool] is not a valid input dtype.";
      }
      break;
    }
    default: {
      MS_LOG(DEBUG) << "For '" << op_name
                    << "', the supported type is in the list: [int32, int64, float32, float64], but got "
                    << x_type->ToString() << ".";
      return nullptr;
    }
  }
  return result;
}

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("ScalarBool", ScalarBoolFrontendFuncImpl);
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("ScalarLog", ScalarLogFrontendFuncImpl);
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("ScalarUadd", ScalarUaddFrontendFuncImpl);
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("ScalarUsub", ScalarUsubFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
