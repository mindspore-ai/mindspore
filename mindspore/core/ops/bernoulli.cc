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

#include "ops/bernoulli.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BernoulliInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_min_shape = shape_map[kMinShape];
  auto input_max_shape = shape_map[kMaxShape];
  auto out_shape = x_shape;
  if (input_min_shape.size() == 0 || input_max_shape.size() == 0) {
    return std::make_shared<abstract::Shape>(out_shape);
  }
  auto output_min_shape = input_min_shape;
  auto output_max_shape = input_max_shape;
  return std::make_shared<abstract::Shape>(out_shape, output_min_shape, output_max_shape);
}

TypePtr BernoulliInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_type = input_args[0]->BuildType();
  const std::set valid_types = {kInt8, kUInt8, kInt16, kInt32, kInt64, kBool, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  auto p_type = input_args[1]->BuildType();
  const std::set p_valid_types = {kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTypeValid("p", p_type, p_valid_types, prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Bernoulli, BaseOperator);
AbstractBasePtr BernoulliInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = BernoulliInferType(primitive, input_args);
  auto infer_shape = BernoulliInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Bernoulli, prim::kPrimBernoulli, BernoulliInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
