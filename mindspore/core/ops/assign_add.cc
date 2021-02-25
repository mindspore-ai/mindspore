/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto assignadd_prim = primitive->cast<PrimAssignAddPtr>();
  MS_EXCEPTION_IF_NULL(assignadd_prim);
  auto prim_name = assignadd_prim->name();
  auto value_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("value_shape", input_args[1]->BuildShape(), prim_name);
  return std::make_shared<abstract::Shape>(value_shape);
}

TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  types.emplace("w", input_args[1]->BuildType());
  // check_scalar_or_tensor_types_same
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, "AssignAdd");
  return TypeIdToType(infer_type);
}
}  // namespace
AbstractBasePtr AssignAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameAssignAdd, AssignAdd);
}  // namespace ops
}  // namespace mindspore
