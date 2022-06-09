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
#include "ops/matrix_band_part.h"
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr MatrixBandPartInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t kInputNums = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNums,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_type = input_args[kInputIndex0]->BuildType();
  const std::set valid_types = {kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("lower", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64},
                                             prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("upper", input_args[kInputIndex2]->BuildType(), {kInt32, kInt64},
                                             prim_name);
  return x_type;
}

abstract::ShapePtr MatrixBandPartInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);

  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto lower_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(lower_shape_ptr);
  auto upper_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(upper_shape_ptr);
  if (x_shape_ptr->IsDynamic() || lower_shape_ptr->IsDynamic() || upper_shape_ptr->IsDynamic()) {
    return x_shape_ptr->cast<abstract::ShapePtr>();
  }

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kGreaterEqual, kXMinShapeSize,
                                           prim_name);
  auto lower_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto upper_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];

  auto broadcast_shape = x_shape;
  if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
    auto expanded_lower_shape = GetExpandedShape<int64_t>(lower_shape);
    // Check whether broadcasting is possible
    (void)CalBroadCastShape(x_shape, expanded_lower_shape, prim_name, "x", "lower");
    // Get broadcast shape
    broadcast_shape = CalBroadCastShape(broadcast_shape, expanded_lower_shape, prim_name);
  }
  if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>()) {
    auto expanded_upper_shape = GetExpandedShape<int64_t>(upper_shape);
    // Check whether broadcasting is possible
    (void)CalBroadCastShape(x_shape, expanded_upper_shape, prim_name, "x", "upper");
    // Get broadcast shape
    broadcast_shape = CalBroadCastShape(broadcast_shape, expanded_upper_shape, prim_name);
  }
  return std::make_shared<abstract::Shape>(broadcast_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(MatrixBandPart, BaseOperator);
AbstractBasePtr MatrixBandPartInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  auto type = MatrixBandPartInferType(primitive, input_args);
  auto shape = MatrixBandPartInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MatrixBandPart, prim::kPrimMatrixBandPart, MatrixBandPartInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
