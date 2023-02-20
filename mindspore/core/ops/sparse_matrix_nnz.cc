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

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
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

  const int kInputNoBatch = 2;
  const int kInputWithBatch = 3;
  const int kZero = 0;
  const int kOne = 1;

  if (!IsDynamicRank(dense_shape) && dense_shape.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_dense_shape should be 1-D, but got " << dense_shape.size()
                             << "-D.";
  }
  if (!IsDynamicRank(batch_pointer) && batch_pointer.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_batch_pointers should be 1-D, but got " << batch_pointer.size()
                             << "-D.";
  }
  if (!IsDynamicRank(row_pointer) && row_pointer.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_row_pointers should be 1-D, but got " << row_pointer.size()
                             << "-D.";
  }
  if (!IsDynamicRank(col_indices) && col_indices.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_col_indices should be 1-D, but got " << col_indices.size()
                             << "-D.";
  }
  if (!IsDynamicRank(values) && values.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, x_values should be 1-D, but got " << values.size() << "-D.";
  }

  const int64_t rank_x = dense_shape[0];
  if (rank_x > kZero && rank_x != kInputNoBatch && rank_x != kInputWithBatch) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, the shape of x_dense_shape must be (2,) or (3,), but got ("
                             << rank_x << ",).";
  }
  if (values[0] > 0 && col_indices[0] > 0 && values[0] != col_indices[0]) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixNNZ, 'x_col_indices' and 'x_values' should have the same length, but "
                                "got length of x_col_indices is "
                             << col_indices[0] << " and length of x_values is " << values[0] << ".";
  }
  std::vector<int64_t> output_shape = {-1};
  if (batch_pointer[0] > 0) {
    output_shape[0] = batch_pointer[0] - 1;
  }
  return std::make_shared<abstract::Shape>(output_shape);
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

// AG means auto generated
class MIND_API AGSparseMatrixNNZInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixNNZInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixNNZInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixNNZInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseMatrixNNZ, prim::kPrimSparseMatrixNNZ, AGSparseMatrixNNZInfer, false);
}  // namespace ops
}  // namespace mindspore
