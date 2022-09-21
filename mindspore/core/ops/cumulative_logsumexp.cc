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
#include "ops/cumulative_logsumexp.h"
#include <set>
#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CumulativeLogsumexpInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  const int64_t min_dim = 1;
  const int64_t kAxisDim = 0;
  (void)CheckAndConvertUtils::CheckInteger("input x rank", SizeToLong(x_shape.size()), kGreaterEqual, min_dim,
                                           prim_name);
  auto axis_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto axis_dim = SizeToLong(axis_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("axis dimension", axis_dim, kEqual, kAxisDim, prim_name);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr CumulativeLogsumexpInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16, kFloat64};
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  const std::set<TypePtr> axis_valid_types = {kInt64, kInt32, kInt16};
  auto axis_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("axis", axis_type, axis_valid_types, prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(CumulativeLogsumexp, BaseOperator);
AbstractBasePtr CumulativeLogsumexpInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto type = CumulativeLogsumexpInferType(primitive, input_args);
  auto shape = CumulativeLogsumexpInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CumulativeLogsumexp, prim::kPrimCumulativeLogsumexp, CumulativeLogsumexpInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
