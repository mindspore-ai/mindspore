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

#include <set>
#include <map>
#include <string>

#include "ops/sparse_matrix_transpose.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseMatrixTransposeInferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kInputNoBatch = 2;
  const int64_t kInputWithBatch = 3;
  const int64_t ktwo = 2;
  std::vector<int64_t> dense_shape_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  const int64_t rank_x = dense_shape_shape[0];
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  const int64_t max_length = GetValue<int64_t>(max_length_ptr);
  std::vector<int64_t> batch_pointers_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  std::vector<int64_t> row_pointers_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  std::vector<int64_t> col_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  std::vector<int64_t> values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  if (rank_x != kInputNoBatch && rank_x != kInputWithBatch) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the rank of input must be 2 or 3, but got "
                             << dense_shape_shape.size() << "!";
  }
  if (batch_pointers_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input batch pointers must be 1-D, but got "
                             << batch_pointers_shape.size() << "-D.";
  }
  if (row_pointers_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input row pointers must be 1-D, but got "
                             << row_pointers_shape.size() << "-D.";
  }
  if (col_indices_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input col indices must be 1-D, but got "
                             << col_indices_shape.size() << "-D.";
  }
  if (values_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input col indices must be 1-D, but got "
                             << col_indices_shape.size() << "-D.";
  }
  auto transpose_shape_shape = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(transpose_shape_shape);
  abstract::ShapePtr transpose_shape_shape_list = transpose_shape_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(transpose_shape_shape_list);
  auto transpose_batch_shape = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(transpose_batch_shape);
  abstract::ShapePtr transpose_batch_shape_list = transpose_batch_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(transpose_batch_shape_list);
  auto transpose_col_indices_shape = input_args[kInputIndex3]->BuildShape();
  MS_EXCEPTION_IF_NULL(transpose_col_indices_shape);
  abstract::ShapePtr transpose_col_indices_shape_list = transpose_col_indices_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(transpose_col_indices_shape_list);
  auto transpose_values_shape = input_args[kInputIndex4]->BuildShape();
  MS_EXCEPTION_IF_NULL(transpose_values_shape);
  abstract::ShapePtr transpose_values_shape_list = transpose_values_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(transpose_values_shape_list);
  if (input_args[kInputIndex0]->isa<abstract::AbstractTensor>() &&
      !input_args[kInputIndex0]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex0]->BuildValue()->isa<None>()) {
    auto dense_shape = input_args[kInputIndex0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(dense_shape);
    auto dense_shape_ptr = dense_shape->BuildValue();
    MS_EXCEPTION_IF_NULL(dense_shape_ptr);
    auto dense_shape_ptr_tensor = CheckAndConvertUtils::CheckTensorIntValue("dense_shape", dense_shape_ptr, prim_name);
    ShapeVector transpose_row_pointers_shape = {0};
    if (rank_x == kInputNoBatch) {
      transpose_row_pointers_shape[0] = dense_shape_ptr_tensor[1] + 1;
    } else {
      transpose_row_pointers_shape[0] = dense_shape_ptr_tensor[0] * (dense_shape_ptr_tensor[ktwo] + 1);
    }
    if (transpose_row_pointers_shape[0] > max_length) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << "the shape of output row pointers must be "
                               << "less than max length: " << max_length << ", but got "
                               << transpose_row_pointers_shape[0]
                               << "! The shape of output row pointers should be reduced"
                               << " or max_length should be increased.";
    }
    ShapeVector transpose_row_pointer_min_shape = {0};
    ShapeVector transpose_row_pointer_max_shape = {max_length};
    abstract::ShapePtr transpose_row_pointers_shape_list = std::make_shared<abstract::Shape>(
      transpose_row_pointers_shape, transpose_row_pointer_min_shape, transpose_row_pointer_max_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      transpose_shape_shape_list, transpose_batch_shape_list, transpose_row_pointers_shape_list,
      transpose_col_indices_shape_list, transpose_values_shape_list});
  } else {
    ShapeVector transpose_row_pointers_shape = {abstract::Shape::SHP_ANY};
    ShapeVector transpose_row_pointer_min_shape = {0};
    ShapeVector transpose_row_pointer_max_shape = {max_length};
    abstract::ShapePtr transpose_row_pointers_shape_list = std::make_shared<abstract::Shape>(
      transpose_row_pointers_shape, transpose_row_pointer_min_shape, transpose_row_pointer_max_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      transpose_shape_shape_list, transpose_batch_shape_list, transpose_row_pointers_shape_list,
      transpose_col_indices_shape_list, transpose_values_shape_list});
  }
}

TuplePtr SparseMatrixTransposeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> index_valid_types = {kInt32, kInt64};
  const std::set<TypePtr> values_valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  auto dense_shape_type = input_args[kInputIndex0]->BuildType();
  auto batch_type = input_args[kInputIndex1]->BuildType();
  auto row_type = input_args[kInputIndex2]->BuildType();
  auto col_type = input_args[kInputIndex3]->BuildType();
  auto value_type = input_args[kInputIndex4]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x_dense_shape", dense_shape_type);
  (void)types.emplace("x_batch_pointers", batch_type);
  (void)types.emplace("x_row_pointers", row_type);
  (void)types.emplace("x_col_indices", col_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, index_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_values", value_type, values_valid_types, prim->name());
  std::vector<TypePtr> types_list = {input_args[kInputIndex0]->BuildType(), input_args[kInputIndex1]->BuildType(),
                                     input_args[kInputIndex2]->BuildType(), input_args[kInputIndex3]->BuildType(),
                                     input_args[kInputIndex4]->BuildType()};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseMatrixTranspose, BaseOperator);
AbstractBasePtr SparseMatrixTransposeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SparseMatrixTransposeInferType(primitive, input_args);
  auto infer_shape = SparseMatrixTransposeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseMatrixTranspose, prim::kPrimSparseMatrixTranspose, SparseMatrixTransposeInfer,
                             nullptr, true);
REGISTER_HOST_DEPENDS(kNameSparseMatrixTranspose, {0});
}  // namespace ops
}  // namespace mindspore
