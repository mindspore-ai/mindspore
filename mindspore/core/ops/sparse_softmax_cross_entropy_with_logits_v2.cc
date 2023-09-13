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

#include "ops/sparse_softmax_cross_entropy_with_logits_v2.h"

#include <map>
#include <memory>
#include <set>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseSoftmaxCrossEntropyWithLogitsV2InferShape(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto features_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto labels_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  const int64_t features_rank = 2;
  const int64_t labels_rank = 1;
  if (!IsDynamic(features_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("dimension of labels", SizeToLong(labels_shape.size()), kEqual,
                                             labels_rank, op_name);
    (void)CheckAndConvertUtils::CheckInteger("dimension of logits(features)", SizeToLong(features_shape.size()), kEqual,
                                             features_rank, op_name);
    (void)CheckAndConvertUtils::CheckInteger("batch of logits(features)", features_shape[kInputIndex0], kEqual,
                                             labels_shape[kInputIndex0], op_name);
  }

  auto loss_shape = labels_shape;
  auto backprop_shape = features_shape;
  abstract::ShapePtr loss_shape_ptr;
  abstract::ShapePtr backprop_shape_ptr;
  loss_shape_ptr = std::make_shared<abstract::Shape>(loss_shape);
  backprop_shape_ptr = std::make_shared<abstract::Shape>(backprop_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{loss_shape_ptr, backprop_shape_ptr});
}

TuplePtr SparseSoftmaxCrossEntropyWithLogitsV2InferType(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto features_type = input_args[kInputIndex0]->BuildType();
  auto labels_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid_features_types = {kFloat16, kFloat32};
  const std::set<TypePtr> valid_labels_types = {kInt32, kInt64};
  std::map<std::string, TypePtr> features_args;
  std::map<std::string, TypePtr> labels_args;
  (void)features_args.emplace("logits_type(features_type)", features_type);
  (void)labels_args.emplace("labels_type", labels_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(features_args, valid_features_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(labels_args, valid_labels_types, primitive->name());

  return std::make_shared<Tuple>(std::vector<TypePtr>{features_type, features_type});
}

AbstractBasePtr SparseSoftmaxCrossEntropyWithLogitsV2Infer(const abstract::AnalysisEnginePtr &,
                                                           const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = SparseSoftmaxCrossEntropyWithLogitsV2InferType(primitive, input_args);
  auto infer_shape = SparseSoftmaxCrossEntropyWithLogitsV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseSoftmaxCrossEntropyWithLogitsV2, BaseOperator);
class MIND_API AGSparseSoftmaxCrossEntropyWithLogitsV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSoftmaxCrossEntropyWithLogitsV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSoftmaxCrossEntropyWithLogitsV2InferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSoftmaxCrossEntropyWithLogitsV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSoftmaxCrossEntropyWithLogitsV2,
                                 prim::kPrimSparseSoftmaxCrossEntropyWithLogitsV2,
                                 AGSparseSoftmaxCrossEntropyWithLogitsV2Infer, false);
}  // namespace ops
}  // namespace mindspore
