/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/generator.h"
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
using namespace generator;
namespace {
int64_t GetCmd(ValuePtr cmd) {
  auto cmd_opt = GetScalarValue<int64_t>(cmd);
  if (MS_UNLIKELY(!cmd_opt.has_value())) {
    MS_LOG(EXCEPTION) << "Cmd value unavailable.";
  }
  auto cmd_value = cmd_opt.value();
  MS_CHECK_VALUE((cmd_value > _START && cmd_value < _END), "Unknown cmd: " + std::to_string(cmd_value));
  return cmd_value;
}

static std::unordered_map<int64_t, std::vector<TypePtr>> kGeneratorInputFormat{
  {STEP, {ParamType, ParamType, ParamType}},         // seed, offset, step
  {SEED, {ParamType, ParamType}},                    // seed, offset
  {GET_STATE, {ParamType, ParamType}},               // seed, offset
  {SET_STATE, {ParamType, ParamType, StateType}},    // seed, offset, state
  {MANUAL_SEED, {ParamType, ParamType, ParamType}},  // seed, offset, new_seed
  {INITIAL_SEED, {ParamType, ParamType}},            // seed, offset
};
}  // namespace
ShapeArray GeneratorFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {{1}, {1}, {sizeof(param_type) * 2}};
}

TypePtrList GeneratorFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {ParamType, ParamType, StateType};
}

int32_t GeneratorFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  int64_t cmd = GetCmd(input_args[kCmdIndex]->GetValue());
  const auto cmd_str = GeneratorEnumToString.at(cmd);
  const auto expected_types = kGeneratorInputFormat[cmd];
  auto input_types = input_args[kInputsIndex]->GetType()->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(input_types);
  if (MS_UNLIKELY(input_types->dynamic_len())) {
    auto element_type = input_types->dynamic_element_type();
    CheckAndConvertUtils::CheckTensorTypeValid("inputs", element_type, {expected_types[0]}, primitive->name());
  } else {
    auto element_types = input_types->elements();
    MS_CHECK_VALUE(element_types.size() == expected_types.size(),
                   "input number for cmd " + cmd_str + " should be " + std::to_string(expected_types.size()) +
                     ", but got " + std::to_string(element_types.size()));
    for (size_t i = 0; i < element_types.size(); ++i) {
      CheckAndConvertUtils::CheckTensorTypeValid("inputs", element_types[i], {expected_types[i]}, primitive->name());
    }
  }
  return true;
}

BaseShapePtr GeneratorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  ShapeArray infer_shapes = {{1}, {1}, {sizeof(param_type) * 2}};
  std::vector<abstract::BaseShapePtr> infer_shape_ptrs(infer_shapes.size());
  std::transform(infer_shapes.begin(), infer_shapes.end(), infer_shape_ptrs.begin(),
                 [](ShapeVector &v) { return std::make_shared<abstract::Shape>(v); });
  return std::make_shared<abstract::TupleShape>(infer_shape_ptrs);
}

TypePtr GeneratorFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  const TypePtrList infer_types = {ParamType, ParamType, StateType};
  return std::make_shared<Tuple>(infer_types);
}

REGISTER_SIMPLE_INFER(kNameGenerator, GeneratorFuncImpl)
}  // namespace mindspore::ops
