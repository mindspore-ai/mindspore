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

#include "ops/sparse_matrix_nnz.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseMatrixNNZInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int64_t> dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  std::vector<int64_t> batch_pointer =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  std::vector<int64_t> row_pointer =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  std::vector<int64_t> col_indices =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  std::vector<int64_t> values =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];

  const int64_t rank_x = dense_shape[0];
  const int kInputNoBatch = 2;
  const int kInputWithBatch = 3;
  const int kOne = 1;
  if (dense_shape.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_dense_shape should be 1-D, bug got " << dense_shape.size()
                             << "-D.";
  }
  if (batch_pointer.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_batch_pointers should be 1-D, bug got " << batch_pointer.size()
                             << "-D.";
  }
  if (row_pointer.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_row_pointers should be 1-D, bug got " << row_pointer.size()
                             << "-D.";
  }
  if (col_indices.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_col_indices should be 1-D, bug got " << col_indices.size()
                             << "-D.";
  }
  if (values.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_values should be 1-D, bug got " << values.size() << "-D.";
  }

  if (!IsDynamic(dense_shape) && rank_x != kInputNoBatch && rank_x != kInputWithBatch) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, the shape of x_dense_shape must be (2,) or (3,), but got ("
                             << rank_x << ",).";
  }
  if (!IsDynamic(values) && !IsDynamic(col_indices) && values[0] != col_indices[0]) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, 'x_col_indices' and 'x_values' should have the same length, bug "
                                "got length of x_col_indices is "
                             << col_indices[0] << " and length of x_values is " << values[0] << ".";
  }
  if (IsDynamic(batch_pointer)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny});
  }
  const int64_t batch_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape][0];
  int64_t y_shape = batch_shape - 1;
  std::vector<int64_t> inShape = {y_shape};
  return std::make_shared<abstract::Shape>(inShape);
}

TypePtr SparseMatrixNNZInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> index_valid_types = {kInt32, kInt64};
  const std::set<TypePtr> values_valid_types = {kInt8,    kInt16,   kInt32,   kInt64,     kUInt8,      kUInt16,
                                                kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};
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
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace

AbstractBasePtr SparseMatrixNNZInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SparseMatrixNNZInferType(primitive, input_args);
  auto infer_shape = SparseMatrixNNZInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseMatrixNNZ, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseMatrixNNZ, prim::kPrimSparseMatrixNNZ, SparseMatrixNNZInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
