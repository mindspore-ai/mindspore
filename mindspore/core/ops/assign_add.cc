/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <map>
#include <string>
#include "ops/assign_add.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AssignAddInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto variable_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto value_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto variable_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(variable_shape_ptr)[kShape];
  auto value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(value_shape_ptr)[kShape];
  auto shape_element = variable_shape_ptr->cast<abstract::ShapePtr>();
  if (variable_shape.size() != value_shape.size()) {
    if (variable_shape.size() == 1 && variable_shape[0] == 1 && value_shape.empty()) {
      return shape_element;
    } else if (value_shape.size() == 1 && value_shape[0] == 1 && variable_shape.empty()) {
      return shape_element;
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', 'value' must have the same rank as 'variable'. But got 'value' rank: "
                               << value_shape.size() << ", 'variable' rank: " << variable_shape.size() << ".";
    }
  }
  for (uint64_t i = 0; i < variable_shape.size(); i++) {
    if (variable_shape[i] != value_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', 'value' must have the same shape as 'variable'. But got 'value' shape: "
                               << value_shape_ptr->ToString()
                               << ", 'variable' shape: " << variable_shape_ptr->ToString() << ".";
    }
  }
  return shape_element;
}

TypePtr AssignAddInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("ref", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("value", input_args[kInputIndex1]->BuildType());
  // check_scalar_or_tensor_types_same
  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex, "AssignAdd");
}
}  // namespace

MIND_API_OPERATOR_IMPL(AssignAdd, BaseOperator);
AbstractBasePtr AssignAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = AssignAddInferType(primitive, input_args);
  auto infer_shape = AssignAddInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape);
}
REGISTER_PRIMITIVE_EVAL_IMPL(AssignAdd, prim::kPrimAssignAdd, AssignAddInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
