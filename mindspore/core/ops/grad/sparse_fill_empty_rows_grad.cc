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

#include "ops/grad/sparse_fill_empty_rows_grad.h"

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseFillEmptyRowsGradInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();

  auto reverse_index_map = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  auto grad_values = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);

  auto reverse_index_map_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(reverse_index_map->BuildShape())[kShape];
  auto grad_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(grad_values->BuildShape())[kShape];

  const int64_t size = 1;
  (void)CheckAndConvertUtils::CheckInteger("reverse_index_map rank", SizeToLong(reverse_index_map_shape.size()), kEqual,
                                           size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("grad_values rank", SizeToLong(grad_values_shape.size()), kEqual, size,
                                           op_name);

  abstract::ShapePtr y_values_shape;
  abstract::ShapePtr y_default_value_shape;

  ShapeVector out_y_values_shape_shape = {reverse_index_map_shape[0]};
  ShapeVector out_y_default_value_shape = {};

  y_values_shape = std::make_shared<abstract::Shape>(out_y_values_shape_shape);
  y_default_value_shape = std::make_shared<abstract::Shape>(out_y_default_value_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{y_values_shape, y_default_value_shape});
}

TypePtr SparseFillEmptyRowsGradInferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> common_valid_types_with_bool_and_complex = {
    kInt8,   kInt16,   kInt32,   kInt64,   kUInt8, kUInt16,    kUInt32,
    kUInt64, kFloat16, kFloat32, kFloat64, kBool,  kComplex64, kComplex128};
  auto reverse_index_map_type = input_args[kInputIndex0]->BuildType();
  auto grad_values_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("reverse_index_map", reverse_index_map_type, {kInt64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad_values", grad_values_type,
                                                   common_valid_types_with_bool_and_complex, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{grad_values_type, grad_values_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseFillEmptyRowsGrad, BaseOperator);
AbstractBasePtr SparseFillEmptyRowsGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = SparseFillEmptyRowsGradInferType(primitive, input_args);
  auto infer_shape = SparseFillEmptyRowsGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseFillEmptyRowsGrad, prim::kPrimSparseFillEmptyRowsGrad, SparseFillEmptyRowsGradInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
