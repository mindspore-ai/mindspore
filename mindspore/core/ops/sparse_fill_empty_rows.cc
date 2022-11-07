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
#include "ops/sparse_fill_empty_rows.h"

#include <set>
#include <string>
#include <map>
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
void CheckSparseFillEmptyRowsInputs(const std::vector<AbstractBasePtr> &input_args, const std::string &op_name) {
  auto indices = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  auto values = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
  auto dense_shape = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 2);
  auto default_value = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 3);

  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(values->BuildShape())[kShape];
  auto dense_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dense_shape->BuildShape())[kShape];
  auto default_value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(default_value->BuildShape())[kShape];

  const int64_t indice_size = 2;
  const int64_t indice_last_dim = 2;
  const int64_t values_size = 1;
  const int64_t dense_shape_size = 1;
  const int64_t default_value_size = 0;

  (void)CheckAndConvertUtils::CheckInteger("indices rank", SizeToLong(indices_shape.size()), kEqual, indice_size,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("values rank", SizeToLong(values_shape.size()), kEqual, values_size,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("dense_shape rank", SizeToLong(dense_shape_shape.size()), kEqual,
                                           dense_shape_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("default_value rank", SizeToLong(default_value_shape.size()), kEqual,
                                           default_value_size, op_name);
  if (indices_shape[1] != indice_last_dim) {
    MS_EXCEPTION(ValueError) << "For SparseFillEmptyRows, "
                             << "the last dim of the indices must be 2, but got " << indices_shape[1];
  }
  if (indices_shape[0] != values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For SparseFillEmptyRows, "
                             << "the indices size must be equal to values first dimension size " << values_shape[0]
                             << ", but got " << indices_shape[0];
  }
}

abstract::TupleShapePtr SparseFillEmptyRowsInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  CheckSparseFillEmptyRowsInputs(input_args, op_name);
  auto indice_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  abstract::ShapePtr output_indices_shape;
  abstract::ShapePtr output_values_shape;
  abstract::ShapePtr output_empty_row_indicator_shape;
  abstract::ShapePtr output_reverse_index_map_shape;

  auto input_shape = input_args[2]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto input_shape_value_ptr = input_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input_shape_value_ptr);

  auto input_type = input_args[2]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_type_id = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type_id);
  auto input_type_element = input_type_id->element();
  MS_EXCEPTION_IF_NULL(input_type_element);
  auto shape_zero = 1;
  if (!input_args[kInputIndex2]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex2]->BuildValue()->isa<None>()) {
    auto input_shape_value_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("x_dense_shape", input_shape_value_ptr, op_name);
    shape_zero = input_shape_value_ptr_tensor[0];
  }
  ShapeVector out_indices_shape = {-1, 2};
  ShapeVector out_values_shape = {-1};
  ShapeVector out_reverse_index_map_shape = {indice_shape[0]};
  ShapeVector min_out_indices_shape = {};
  ShapeVector max_out_indices_shape = {};
  ShapeVector min_out_values_shape = {};
  ShapeVector max_out_values_shape = {};
  ShapeVector out_empty_row_indicator_shape = {shape_zero};

  min_out_indices_shape.push_back(indice_shape[0]);
  min_out_indices_shape.push_back(indice_shape[1]);
  min_out_values_shape.push_back(indice_shape[0]);

  max_out_indices_shape.push_back(indice_shape[0] + shape_zero);
  max_out_indices_shape.push_back(indice_shape[1]);
  max_out_values_shape.push_back(indice_shape[0] + shape_zero);

  output_indices_shape =
    std::make_shared<abstract::Shape>(out_indices_shape, min_out_indices_shape, max_out_indices_shape);
  output_values_shape = std::make_shared<abstract::Shape>(out_values_shape, min_out_values_shape, max_out_values_shape);
  output_empty_row_indicator_shape = std::make_shared<abstract::Shape>(out_empty_row_indicator_shape);
  output_reverse_index_map_shape = std::make_shared<abstract::Shape>(out_reverse_index_map_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    output_indices_shape, output_values_shape, output_empty_row_indicator_shape, output_reverse_index_map_shape});
}

TypePtr SparseFillEmptyRowsInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> common_valid_types_with_bool_and_complex = {
    kInt8,   kInt16,   kInt32,   kInt64,   kUInt8, kUInt16,    kUInt32,
    kUInt64, kFloat16, kFloat32, kFloat64, kBool,  kComplex64, kComplex128};
  auto indices_type = input_args[kInputIndex0]->BuildType();
  auto values_type = input_args[kInputIndex1]->BuildType();
  auto dense_shape_type = input_args[kInputIndex2]->BuildType();
  auto default_value_type = input_args[kInputIndex3]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("values", values_type);
  (void)types.emplace("default_value", default_value_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_bool_and_complex, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, {kInt64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("dense_shape", dense_shape_type, {kInt64}, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{std::make_shared<TensorType>(kInt64), values_type,
                                                      std::make_shared<TensorType>(kBool),
                                                      std::make_shared<TensorType>(kInt64)});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseFillEmptyRows, BaseOperator);
AbstractBasePtr SparseFillEmptyRowsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = SparseFillEmptyRowsInferType(primitive, input_args);
  auto infer_shape = SparseFillEmptyRowsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_HOST_DEPENDS(kNameSparseFillEmptyRows, {2});
REGISTER_PRIMITIVE_EVAL_IMPL(SparseFillEmptyRows, prim::kPrimSparseFillEmptyRows, SparseFillEmptyRowsInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
