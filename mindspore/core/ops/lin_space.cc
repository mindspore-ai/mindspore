/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/lin_space.h"
#include <memory>
#include <map>
#include <string>
#include "ops/primitive_c.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr LinSpaceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractScalar>(prim_name, input_args, kInputIndex2);

  auto num_value = input_args[kInputIndex2]->BuildValue();
  MS_EXCEPTION_IF_NULL(num_value);
  if (!num_value->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the 'num' must be a Int, but got "
                            << num_value->ToString();
  }

  auto start_dtype = input_args[kInputIndex0]->BuildType();
  auto stop_dtype = input_args[kInputIndex1]->BuildType();

  std::map<std::string, TypePtr> type_dict = {
    {"start type", start_dtype},
    {"stop type", stop_dtype},
  };
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, {kFloat32}, prim_name);
}
abstract::ShapePtr LinSpaceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();

  auto start_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(start_shape_ptr);
  auto stop_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(stop_shape_ptr);

  // Do it later
  if (start_shape_ptr->IsDynamic() || stop_shape_ptr->IsDynamic()) {
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }

  const auto start_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(start_shape_ptr)[kShape];
  const auto stop_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(stop_shape_ptr)[kShape];

  size_t start_dims = start_shape.size();
  size_t stop_dims = stop_shape.size();

  CheckAndConvertUtils::CheckValue<size_t>("dimension of 'start'", start_dims, kEqual, "dimension of 'stop'", stop_dims,
                                           prim_name);

  for (size_t i = 0; i < start_dims; ++i) {
    CheckAndConvertUtils::CheckValue<int64_t>(std::to_string(i) + "th dimension of 'start'", start_shape[i], kEqual,
                                              std::to_string(i) + "th dimension of 'stop'", stop_shape[i], prim_name);
  }

  // Checked in LinSpaceInferType, num is a Scalar
  const auto num_value = input_args[kInputIndex2]->BuildValue();
  const int64_t num = num_value->cast<Int64ImmPtr>()->value();

  CheckAndConvertUtils::CheckValue<int64_t>("num", num, kGreaterEqual, 0, prim_name);

  ShapeVector out_shape(start_shape.begin(), start_shape.end());
  out_shape.push_back(num);

  return std::make_shared<abstract::Shape>(out_shape);
}
}  // namespace
MIND_API_OPERATOR_IMPL(LinSpace, BaseOperator);
AbstractBasePtr LinSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = LinSpaceInferType(primitive, input_args);
  auto infer_shape = LinSpaceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(LinSpace, prim::kPrimLinSpace, LinSpaceInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
