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
#include <string>
#include <vector>
#include <memory>

#include "ops/csr_sparse_matrix_to_sparse_tensor.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
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
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TuplePtr CSRSparseMatrixToSparseTensorInferType(const PrimitivePtr &prim,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_dense_shape_type = input_args[kInputIndex0]->BuildType();
  auto x_batch_pointers_type = input_args[kInputIndex1]->BuildType();
  auto x_row_pointers_type = input_args[kInputIndex2]->BuildType();
  auto x_col_indices_type = input_args[kInputIndex3]->BuildType();
  auto x_values_type = input_args[kInputIndex4]->BuildType();
  const std::set<TypePtr> common_valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dense_shape", x_dense_shape_type, {kInt32, kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_batch_pointers", x_batch_pointers_type, {kInt32, kInt64},
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_row_pointers", x_row_pointers_type, {kInt32, kInt64},
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_col_indices", x_col_indices_type, {kInt32, kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_values", x_values_type, common_valid_types, prim->name());
  std::vector<TypePtr> types_list = {input_args[0]->BuildType(), input_args[4]->BuildType(),
                                     input_args[0]->BuildType()};
  return std::make_shared<Tuple>(types_list);
}

abstract::TupleShapePtr CSRSparseMatrixToSparseTensorInferShape(const PrimitivePtr &primitive,
                                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kZero = 0;
  const int64_t kOne = 1;
  const int64_t kDefalutRank = 2;
  const int64_t kBatchRank = 3;
  CheckInputShapeEmpty(primitive->name(), input_args);
  std::vector<int64_t> x_dense_shape_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto prim_name = primitive->name();
  auto x_batch_pointers_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto x_row_pointers_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto x_col_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto x_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  const int64_t x_dense_shape_rank = x_dense_shape_shape.size();
  const int64_t x_batch_pointers_rank = x_batch_pointers_shape.size();
  const int64_t x_row_pointers_rank = x_row_pointers_shape.size();
  const int64_t x_col_indices_rank = x_col_indices_shape.size();
  const int64_t x_values_rank = x_values_shape.size();
  if (x_dense_shape_rank != kOne || x_batch_pointers_rank != kOne || x_row_pointers_rank != kOne ||
      x_col_indices_rank != kOne || x_values_rank != kOne) {
    MS_EXCEPTION(ValueError) << "For CSRSparseMatrixToSparseTensor, input x_dense_shape should be a 1-D tensor"
                             << ", but got " << x_dense_shape_shape.size() << "-D"
                             << ", input x_batch_pointers should be a 1-D tensor"
                             << ", but got " << x_batch_pointers_shape.size() << "-D"
                             << ", input x_row_pointers should be a 1-D tensor"
                             << ", but got " << x_row_pointers_shape.size() << "-D"
                             << ", input x_col_indices should be a 1-D tensor"
                             << ", but got " << x_col_indices_shape.size() << "-D"
                             << ", input x_values should be a 1-D tensor"
                             << ", but got " << x_values_shape.size() << "-D";
  }
  if (!IsDynamic(x_col_indices_shape) && !IsDynamic(x_values_shape)) {
    if (x_col_indices_shape[kZero] != x_values_shape[kZero]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", x_col_indices.shape[0] and x_values.shape[0] should be the same"
                               << ", but got x_col_indices.shape[0] = " << x_col_indices_shape[kZero]
                               << ", x_values.shape[0] = " << x_values_shape[kZero];
    }
  }
  int64_t rank_x = x_dense_shape_shape[kZero];
  if (!IsDynamic(x_dense_shape_shape)) {
    if (rank_x != kDefalutRank && rank_x != kBatchRank) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input dense_shape should "
                               << "have rank 2 or 3, but got " << rank_x << ".";
    }
  } else {
    rank_x = abstract::Shape::kShapeDimAny;
  }
  auto x_values_shape_val =
    x_values_shape[kZero] != abstract::Shape::kShapeRankAny ? x_values_shape[kZero] : abstract::Shape::kShapeDimAny;
  ShapeVector indices_shape = {x_values_shape_val, rank_x};
  ShapeVector values_shape = {x_values_shape_val};
  ShapeVector dense_shape_shape = {rank_x};
  std::vector<BaseShapePtr> shapes_list;
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(indices_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(values_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(dense_shape_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}
}  // namespace

MIND_API_OPERATOR_IMPL(CSRSparseMatrixToSparseTensor, BaseOperator);
AbstractBasePtr CSRSparseMatrixToSparseTensorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = CSRSparseMatrixToSparseTensorInferType(primitive, input_args);
  auto infer_shape = CSRSparseMatrixToSparseTensorInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGCSRSparseMatrixToSparseTensorInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CSRSparseMatrixToSparseTensorInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CSRSparseMatrixToSparseTensorInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CSRSparseMatrixToSparseTensorInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CSRSparseMatrixToSparseTensor, prim::kPrimCSRSparseMatrixToSparseTensor,
                                 AGCSRSparseMatrixToSparseTensorInfer, false);
}  // namespace ops
}  // namespace mindspore
