/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/sigmoid_cross_entropy_with_logits.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SigmoidCrossEntropyWithLogitsInferShape(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_LOG(INFO) << "For '" << op_name << "', it's now doing infer shape.";
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, op_name);
  auto logits_shape = input_args[0]->BuildShape();
  auto label_shape = input_args[1]->BuildShape();
  auto logits_shape_ptr = logits_shape->cast<abstract::ShapePtr>();
  auto label_shape_ptr = label_shape->cast<abstract::ShapePtr>();
  // logits and label must have the same shape when is not dynamic
  if (!logits_shape_ptr->IsDynamic() && !label_shape_ptr->IsDynamic()) {
    if (*logits_shape != *label_shape) {
      MS_EXCEPTION(ValueError)
        << "For " << op_name
        << ", evaluator arg label shape should be consistent with logits shape, but got label shape: "
        << label_shape->ToString() << ", logits shape: " << logits_shape->ToString();
    }
  }
  auto logits_element = logits_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(logits_element);
  return logits_element;
}

TypePtr SigmoidCrossEntropyWithLogitsInferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto logits_type = input_args[kInputIndex0]->BuildType();
  auto label_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid_types = {kBool,   kInt,    kInt8,   kInt16, kInt32,   kInt64,   kUInt,    kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64};
  std::map<std::string, TypePtr> args;
  (void)args.insert({"logits_type", logits_type});
  (void)args.insert({"label_type", label_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, primitive->name());
  return logits_type;
}
}  // namespace

MIND_API_BASE_IMPL(SigmoidCrossEntropyWithLogits, PrimitiveC, BaseOperator);
AbstractBasePtr SigmoidCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = SigmoidCrossEntropyWithLogitsInferType(primitive, input_args);
  auto infer_shape = SigmoidCrossEntropyWithLogitsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SigmoidCrossEntropyWithLogits, prim::kPrimSigmoidCrossEntropyWithLogits,
                             SigmoidCrossEntropyWithLogitsInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
