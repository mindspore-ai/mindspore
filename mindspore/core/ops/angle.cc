/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/angle.h"
#include <string>
#include <algorithm>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AngleInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto primitive_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(primitive_name, input_args, 0);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr AngleInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto primitive_name = primitive->name();
  auto input_type = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types, primitive_name);
  auto input_tensor = input_type->cast<TensorTypePtr>();
  TypeId input_tensor_id = input_tensor->element()->type_id();
  return input_tensor_id == kNumberTypeComplex64 ? std::make_shared<TensorType>(kFloat32)
                                                 : std::make_shared<TensorType>(kFloat64);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Angle, BaseOperator);
AbstractBasePtr AngleInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto primitive_name = primitive->name();
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto infer_type = AngleInferType(primitive, input_args);
  auto infer_shape = AngleInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Angle, prim::kPrimAngle, AngleInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
