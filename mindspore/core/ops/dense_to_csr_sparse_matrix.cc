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

#include "ops/dense_to_csr_sparse_matrix.h"

#include <map>
#include <memory>
#include <set>

#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/shape_utils.h"
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
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr DenseToCSRSparseMatrixInferShape(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  CheckInputShapeEmpty(primitive->name(), input_args);
  auto dense_input_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto dense_input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dense_input_shape_ptr);
  auto dense_input_shape = dense_input_shape_map[kShape];
  auto indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto indices_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr);
  auto indices_shape = indices_shape_map[kShape];
  const int64_t kZero = 0;
  const int64_t kOne = 1;
  const int64_t kIndicesRank = 2;
  const int64_t kDefalutRank = 2;
  const int64_t kBatchRank = 3;
  const int64_t rank = SizeToLong(dense_input_shape.size());
  const int64_t indices_rank = SizeToLong(indices_shape.size());
  auto dense_input_is_dynamic_rank = IsDynamicRank(dense_input_shape);
  if (rank != kDefalutRank && rank != kBatchRank && !dense_input_is_dynamic_rank) {
    MS_EXCEPTION(ValueError) << "Currently only support 2-D Tensor when converting to CSRTensor,"
                             << " but got Tensor dimension: " << rank << ".";
  }
  if (indices_rank != kIndicesRank && !IsDynamicRank(indices_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', indices should "
                             << "have rank 2, but got " << indices_rank << ".";
  }
  if (!IsDynamic(indices_shape) && rank != indices_shape[kOne] && !dense_input_is_dynamic_rank) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', shape[1] of indices must be equal "
                             << "to the rank of dense input, but got dense rank: " << rank << ", "
                             << "indices.shape[1]: " << indices_shape[kOne] << ".";
  }
  const int64_t batch_size = (rank == kDefalutRank) ? kOne : dense_input_shape[kZero];
  int64_t num_rows = abstract::Shape::kShapeDimAny;
  if (!dense_input_is_dynamic_rank) {
    num_rows = (rank == kDefalutRank) ? dense_input_shape[kZero] : dense_input_shape[kOne];
  }
  auto y_dense_shape_shape_val = dense_input_is_dynamic_rank ? abstract::Shape::kShapeDimAny : rank;
  ShapeVector y_dense_shape_shape{y_dense_shape_shape_val};
  auto y_batch_pointers_shape_val = batch_size >= 0 ? batch_size + kOne : abstract::Shape::kShapeDimAny;
  ShapeVector y_batch_pointers_shape{y_batch_pointers_shape_val};
  auto y_row_pointers_shape_val = abstract::Shape::kShapeDimAny;
  if (batch_size >= 0 && num_rows >= 0) {
    y_row_pointers_shape_val = batch_size * (num_rows + kOne);
  }
  ShapeVector y_row_pointers_shape{y_row_pointers_shape_val};
  ShapeVector y_col_indices_shape{indices_shape[kZero]};
  ShapeVector y_values_shape{indices_shape[kZero]};
  std::vector<BaseShapePtr> shapes_list;
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(y_dense_shape_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(y_batch_pointers_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(y_row_pointers_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(y_col_indices_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(y_values_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TuplePtr DenseToCSRSparseMatrixInferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> values_types = {kFloat64, kFloat32, kComplex128, kComplex64};
  const std::set<TypePtr> indices_types = {kInt32, kInt64};
  auto dense_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("dense_input", dense_type, values_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices_type", indices_type, indices_types, op_name);
  std::vector<TypePtr> types_list{indices_type, indices_type, indices_type, indices_type, dense_type};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace

MIND_API_OPERATOR_IMPL(DenseToCSRSparseMatrix, BaseOperator);
AbstractBasePtr DenseToCSRSparseMatrixInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = DenseToCSRSparseMatrixInferType(primitive, input_args);
  auto shapes = DenseToCSRSparseMatrixInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGDenseToCSRSparseMatrixInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DenseToCSRSparseMatrixInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DenseToCSRSparseMatrixInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DenseToCSRSparseMatrixInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DenseToCSRSparseMatrix, prim::kPrimDenseToCSRSparseMatrix,
                                 AGDenseToCSRSparseMatrixInfer, false);
}  // namespace ops
}  // namespace mindspore
