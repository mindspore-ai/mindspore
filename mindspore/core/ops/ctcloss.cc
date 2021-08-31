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

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
void CheckCTCLossInputs(const std::vector<AbstractBasePtr> &input_args, const std::string &op_name) {
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

  const int64_t input_size = 3;
  const int64_t label_indice_size = 2;
  const int64_t label_indice_last_dim = 2;
  (void)CheckAndConvertUtils::CheckInteger("inputs rank", SizeToLong(inputs_shape.size()), kEqual, input_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("label_indices rank", SizeToLong(labels_indices_shape.size()), kEqual,
                                           label_indice_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("label_indices second dim", labels_indices_shape[1], kEqual,
                                           label_indice_last_dim, op_name);
  (void)CheckAndConvertUtils::CheckInteger("label_values rank", int64_t(labels_values_shape.size()), kEqual, 1,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("sequence_length rank", int64_t(sequence_length_shape.size()), kEqual, 1,
                                           op_name);

  if (labels_indices_shape[0] != labels_values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For CTCLoss first dim of label_indices and label_value must be same, but got "
                             << labels_indices_shape[0] << " and " << labels_values_shape[0];
  }
  if (inputs_shape[1] != sequence_length_shape[0]) {
    MS_EXCEPTION(ValueError) << "For CTCLoss input batch_size must be same with sequence_length batch_size, but got "
                             << inputs_shape[1] << " and " << sequence_length_shape[0];
  }
}

abstract::TupleShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  CheckCTCLossInputs(input_args, op_name);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto shape = input_shape[kShape];
  auto min_shape = input_shape[kMinShape];
  auto max_shape = input_shape[kMaxShape];

  ShapeVector batch = {shape[1]};
  abstract::ShapePtr loss_shape;
  abstract::ShapePtr gradient_shape;
  if (min_shape.empty() || max_shape.empty()) {
    loss_shape = std::make_shared<abstract::Shape>(batch);
    gradient_shape = std::make_shared<abstract::Shape>(shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{loss_shape, gradient_shape});
  }

  ShapeVector batch_min = {min_shape[1]};
  ShapeVector batch_max = {max_shape[1]};
  loss_shape = std::make_shared<abstract::Shape>(batch, batch_min, batch_max);
  gradient_shape = std::make_shared<abstract::Shape>(shape, min_shape, max_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{loss_shape, gradient_shape});
}

TuplePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
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

AbstractBasePtr CTCLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto types = InferType(primitive, input_args);
  auto shapes = InferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CTCLoss, prim::kPrimCTCLoss, CTCLossInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
