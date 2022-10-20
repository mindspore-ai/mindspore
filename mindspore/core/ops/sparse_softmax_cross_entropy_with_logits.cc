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

#include <set>
#include <map>
#include <string>

#include "ops/sparse_softmax_cross_entropy_with_logits.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseSoftmaxCrossEntropyWithLogitsInferShape(const PrimitivePtr &primitive,
                                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto features_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto labels_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  const int64_t features_rank = 2;
  const int64_t labels_rank = 1;
  if (!IsDynamic(features_shape) && !IsDynamic(labels_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("batch of logits(features)", features_shape[kInputIndex0], kEqual,
                                             labels_shape[kInputIndex0], op_name);
    (void)CheckAndConvertUtils::CheckInteger("dimension of logits(features)", SizeToLong(features_shape.size()), kEqual,
                                             features_rank, op_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("dimension of labels", SizeToLong(labels_shape.size()), kEqual, labels_rank,
                                           op_name);
  auto is_grad = false;
  if (primitive->HasAttr(kIsGrad)) {
    is_grad = GetValue<bool>(primitive->GetAttr(kIsGrad));
  }
  ShapeVector output_shape{};
  if (is_grad) {
    output_shape = features_shape;
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr SparseSoftmaxCrossEntropyWithLogitsInferType(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto features_type = input_args[kInputIndex0]->BuildType();
  auto labels_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid_features_types = {kFloat16, kFloat32};
  const std::set<TypePtr> valid_labels_types = {kInt32, kInt64};
  std::map<std::string, TypePtr> features_args, labels_args;
  (void)features_args.insert({"logits_type(features_type)", features_type});
  (void)labels_args.insert({"labels_type", labels_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(features_args, valid_features_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(labels_args, valid_labels_types, primitive->name());
  return features_type;
}
}  // namespace

void SparseSoftmaxCrossEntropyWithLogits::Init(const bool is_grad) { this->set_is_grad(is_grad); }

void SparseSoftmaxCrossEntropyWithLogits::set_is_grad(const bool is_grad) {
  (void)this->AddAttr(kIsGrad, api::MakeValue(is_grad));
}

AbstractBasePtr SparseSoftmaxCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = SparseSoftmaxCrossEntropyWithLogitsInferType(primitive, input_args);
  auto infer_shape = SparseSoftmaxCrossEntropyWithLogitsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool SparseSoftmaxCrossEntropyWithLogits::get_is_grad() const { return GetValue<bool>(GetAttr(kIsGrad)); }

MIND_API_OPERATOR_IMPL(SparseSoftmaxCrossEntropyWithLogits, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseSoftmaxCrossEntropyWithLogits, prim::kPrimSparseSoftmaxCrossEntropyWithLogits,
                             SparseSoftmaxCrossEntropyWithLogitsInfer, nullptr, true);
// REGISTER_PRIMITIVE_C(kNameSparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogits);
}  // namespace ops
}  // namespace mindspore
