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

#include "ops/softmax_cross_entropy_with_logits.h"
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SoftmaxCrossEntropyWithLogitsInferShape(const PrimitivePtr &primitive,
                                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto logits_shape = input_args[0]->BuildShape();
  auto label_shape = input_args[1]->BuildShape();
  auto logits_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(logits_shape)[kShape];
  auto label_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(label_shape)[kShape];
  const int64_t input_rank = 2;
  (void)CheckAndConvertUtils::CheckInteger("dimension of logits", SizeToLong(logits_map.size()), kEqual, input_rank,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of labels", SizeToLong(label_map.size()), kEqual, input_rank,
                                           prim_name);
  auto logits_shape_ptr = logits_shape->cast<abstract::ShapePtr>();
  auto label_shape_ptr = label_shape->cast<abstract::ShapePtr>();
  // logits and label must have the same shape when is not dynamic
  if (!logits_shape_ptr->IsDynamic() && !label_shape_ptr->IsDynamic()) {
    if (*logits_shape != *label_shape) {
      MS_EXCEPTION(ValueError) << prim_name << " evaluator arg label shape " << label_shape->ToString()
                               << " are not consistent with logits shape " << logits_shape->ToString();
    }
  }
  auto logits_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(logits_shape);
  auto logits_shp = logits_shape_map[kShape];
  ShapeVector logits_max_shape = logits_shape_map[kMaxShape];
  ShapeVector logits_min_shape = logits_shape_map[kMinShape];
  ShapeVector loss_shape = {logits_shp[0]};
  abstract::ShapePtr loss_shape_ptr;
  if (logits_shape_ptr->IsDynamic()) {
    ShapeVector loss_min_shape = {logits_min_shape[0]};
    ShapeVector loss_max_shape = {logits_max_shape[0]};
    loss_shape_ptr = std::make_shared<abstract::Shape>(loss_shape, loss_min_shape, loss_max_shape);
  } else {
    loss_shape_ptr = std::make_shared<abstract::Shape>(loss_shape);
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{loss_shape_ptr, logits_shape_ptr});
}

TuplePtr SoftmaxCrossEntropyWithLogitsInferType(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto logits_type = input_args[0]->BuildType();
  auto label_type = input_args[1]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> args;
  (void)args.insert({"logits_type", logits_type});
  (void)args.insert({"label_type", label_type});
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, primitive->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type});
}
}  // namespace
AbstractBasePtr SoftmaxCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = SoftmaxCrossEntropyWithLogitsInferType(primitive, input_args);
  auto infer_shape = SoftmaxCrossEntropyWithLogitsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SoftmaxCrossEntropyWithLogits, prim::kPrimSoftmaxCrossEntropyWithLogits,
                             SoftmaxCrossEntropyWithLogitsInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
