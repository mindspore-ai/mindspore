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
#include "ops/ctcloss.h"

#include <set>
#include <string>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/shape_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void CheckCTCLossInputs(const ShapeVector &inputs_shape, const ShapeVector &labels_indices_shape,
                        const ShapeVector &labels_values_shape, const ShapeVector &sequence_length_shape,
                        const std::string &op_name) {
  const int64_t input_size = 3;
  const int64_t label_indice_size = 2;
  const int64_t label_indice_last_dim = 2;
  (void)CheckAndConvertUtils::CheckInteger("inputs rank", SizeToLong(inputs_shape.size()), kEqual, input_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("label_indices rank", SizeToLong(labels_indices_shape.size()), kEqual,
                                           label_indice_size, op_name);
  if (labels_indices_shape[1] != abstract::Shape::kShapeDimAny) {
    (void)CheckAndConvertUtils::CheckInteger("label_indices second dim", labels_indices_shape[1], kEqual,
                                             label_indice_last_dim, op_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("label_values rank", int64_t(labels_values_shape.size()), kEqual, 1,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("sequence_length rank", int64_t(sequence_length_shape.size()), kEqual, 1,
                                           op_name);

  if (labels_indices_shape[0] != abstract::Shape::kShapeDimAny &&
      labels_values_shape[0] != abstract::Shape::kShapeDimAny && labels_indices_shape[0] != labels_values_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For 'CTCLoss', the first dim of 'label_indices' and 'label_value' must be same, but got 'label_indices':"
      << labels_indices_shape[0] << ", 'label_value': " << labels_values_shape[0] << ".";
  }
  if (inputs_shape[1] != abstract::Shape::kShapeDimAny && sequence_length_shape[0] != abstract::Shape::kShapeDimAny &&
      inputs_shape[1] != sequence_length_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For 'CTCLoss', input batch_size must be same with 'sequence_length' batch_size, but got input batch_size:"
      << inputs_shape[1] << ", 'sequence_length' batch_size: " << sequence_length_shape[0] << ".";
  }
}

abstract::TupleShapePtr CTCLossInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t input_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           op_name);
  auto inputs = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  auto labels_indices = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
  auto labels_values = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 2);
  auto sequence_length = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 3);
  auto inputs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs->BuildShape())[kShape];
  auto labels_indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(labels_indices->BuildShape())[kShape];
  auto labels_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(labels_values->BuildShape())[kShape];
  auto sequence_length_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(sequence_length->BuildShape())[kShape];
  if (IsDynamicRank(inputs_shape) || IsDynamicRank(labels_indices_shape) || IsDynamicRank(labels_values_shape) ||
      IsDynamicRank(sequence_length_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny}),
      std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny})});
  }
  CheckCTCLossInputs(inputs_shape, labels_indices_shape, labels_values_shape, sequence_length_shape, op_name);

  ShapeVector batch = {inputs_shape[1]};
  abstract::ShapePtr loss_shape;
  abstract::ShapePtr gradient_shape;
  loss_shape = std::make_shared<abstract::Shape>(batch);
  gradient_shape = std::make_shared<abstract::Shape>(inputs_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{loss_shape, gradient_shape});
}

TuplePtr CTCLossInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("labels_indices", input_args[kInputIndex1]->BuildType(), {kInt64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("labels_values", input_args[kInputIndex2]->BuildType(), {kInt32},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("sequence_length", input_args[kInputIndex3]->BuildType(), {kInt32},
                                                   op_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto type =
    CheckAndConvertUtils::CheckTensorTypeValid("inputs", input_args[kInputIndex0]->BuildType(), valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(CTCLoss, BaseOperator);
AbstractBasePtr CTCLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto types = CTCLossInferType(primitive, input_args);
  auto shapes = CTCLossInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGCTCLossInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CTCLoss, prim::kPrimCTCLoss, AGCTCLossInfer, false);
}  // namespace ops
}  // namespace mindspore
