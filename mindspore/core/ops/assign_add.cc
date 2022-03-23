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
  auto x_shape = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto output_shape = x_shape->cast<abstract::ShapePtr>();
  return output_shape;
}

TypePtr AssignAddInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("ref", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("value", input_args[kInputIndex1]->BuildType());
  // check_scalar_or_tensor_types_same
  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, "AssignAdd");
}
}  // namespace

MIND_API_BASE_IMPL(AssignAdd, PrimitiveC, BaseOperator);
AbstractBasePtr AssignAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = AssignAddInferType(primitive, input_args);
  auto infer_shape = AssignAddInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape);
}
REGISTER_PRIMITIVE_EVAL_IMPL(AssignAdd, prim::kPrimAssignAdd, AssignAddInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
