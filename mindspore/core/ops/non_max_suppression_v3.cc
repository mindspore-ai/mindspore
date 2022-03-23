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

#include <set>

#include "ops/non_max_suppression_v3.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NonMaxSuppressionV3InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(primitive);
  const int input_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  auto boxes_shape = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape]);
  auto scores_shape = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack())[kShape]);
  auto max_output_size_shape = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShapeTrack())[kShape]);
  auto iou_threshold_shape = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->GetShapeTrack())[kShape]);
  auto score_threshold_shape = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShapeTrack())[kShape]);
  // boxes second dimension must euqal 4
  (void)CheckAndConvertUtils::CheckInteger("boxes second dimension", boxes_shape->shape()[1], kEqual, 4, prim_name);
  // boxes must be rank 2
  (void)CheckAndConvertUtils::CheckInteger("boxes rank", boxes_shape->shape().size(), kEqual, 2, prim_name);
  // score must be rank 1
  (void)CheckAndConvertUtils::CheckInteger("scores rank", scores_shape->shape().size(), kEqual, 1, prim_name);
  // score length must be equal with boxes first dimension
  (void)CheckAndConvertUtils::CheckInteger("scores length", scores_shape->shape()[0], kEqual, boxes_shape->shape()[0],
                                           prim_name);
  // max_output_size,iou_threshold,score_threshold must be scalar
  (void)CheckAndConvertUtils::CheckInteger("max_output_size size", max_output_size_shape->shape().size(), kEqual, 0,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("iou_threshold size", iou_threshold_shape->shape().size(), kEqual, 0,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("score_threshold size", score_threshold_shape->shape().size(), kEqual, 0,
                                           prim_name);
  auto scores_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  // calculate output shape
  ShapeVector selected_indices_shape = {abstract::Shape::SHP_ANY};
  ShapeVector selected_indices_min_shape = {0};
  ShapeVector selected_indices_max_shape;
  if (scores_shape_map[kShape].size() > 0 && scores_shape_map[kShape][0] == -1) {
    selected_indices_max_shape = scores_shape_map[kMaxShape];
    return std::make_shared<abstract::Shape>(selected_indices_shape, selected_indices_min_shape,
                                             selected_indices_max_shape);
  }
  selected_indices_max_shape = scores_shape_map[kShape];
  return std::make_shared<abstract::Shape>(selected_indices_shape, selected_indices_min_shape,
                                           selected_indices_max_shape);
}

TypePtr NonMaxSuppressionV3InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(prim);
  const int input_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto boxes_type = input_args[0]->BuildType();
  auto scores_type = input_args[1]->BuildType();
  auto max_output_size_type = input_args[2]->BuildType();
  auto iou_threshold_type = input_args[3]->BuildType();
  auto score_threshold_type = input_args[4]->BuildType();
  // boxes and scores must have same type
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> args;
  args.insert({"boxes_type", boxes_type});
  args.insert({"scores_type", scores_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // iou_threshold,score_threshold must be scalar
  std::map<std::string, TypePtr> args2;
  args2.insert({"iou_threshold_type", iou_threshold_type});
  args2.insert({"score_threshold_type", score_threshold_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args2, valid_types, prim_name);
  // max_output_size must be scalar
  const std::set<TypePtr> valid_types2 = {kInt32, kInt64};
  std::map<std::string, TypePtr> args3;
  args3.insert({"max_output_size_type", max_output_size_type});
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args3, valid_types2, prim_name);
  return max_output_size_type;
}
}  // namespace

MIND_API_BASE_IMPL(NonMaxSuppressionV3, PrimitiveC, BaseOperator);
AbstractBasePtr NonMaxSuppressionV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(NonMaxSuppressionV3InferShape(primitive, input_args),
                                NonMaxSuppressionV3InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(NonMaxSuppressionV3, prim::kPrimNonMaxSuppressionV3, NonMaxSuppressionV3Infer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
