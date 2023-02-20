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
#include <vector>
#include <memory>
#include <algorithm>

#include "ops/sparse_tensor_to_csr_sparse_matrix.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/value.h"
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
TuplePtr SparseTensorToCSRSparseMatrixInferType(const PrimitivePtr &prim,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_indices_type = input_args[kInputIndex0]->BuildType();
  auto x_values_type = input_args[kInputIndex1]->BuildType();
  auto x_dense_shape_type = input_args[kInputIndex2]->BuildType();
  const std::set<TypePtr> common_valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_indices", x_indices_type, {kInt32, kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_values", x_values_type, common_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dense_shape", x_dense_shape_type, {kInt32, kInt64}, prim->name());
  std::vector<TypePtr> types_list = {input_args[kInputIndex2]->BuildType(), input_args[kInputIndex0]->BuildType(),
                                     input_args[kInputIndex0]->BuildType(), input_args[kInputIndex0]->BuildType(),
                                     input_args[kInputIndex1]->BuildType()};
  return std::make_shared<Tuple>(types_list);
}

abstract::TupleShapePtr SparseTensorToCSRSparseMatrixInferShape(const PrimitivePtr &primitive,
                                                                const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kIndicesRank = 2;
  const int64_t kDefalutRank = 2;
  const int64_t kBatchRank = 3;
  std::vector<int64_t> x_dense_shape_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (x_dense_shape_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input x_dense_shape should "
                             << "have rank 2 or 3, but got " << x_dense_shape_shape.size() << ".";
  }
  const int64_t rank_x = x_dense_shape_shape[0];
  if (!IsDynamic(x_dense_shape_shape) && rank_x != kDefalutRank && rank_x != kBatchRank) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input x_dense_shape should "
                             << "have rank 2 or 3, but got " << rank_x << ".";
  }
  auto prim_name = primitive->name();
  auto x_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  std::vector<ShapeVector> all_shapes = {x_indices_shape, x_values_shape, x_dense_shape_shape};
  auto is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);
  const int64_t x_indices_rank = static_cast<int64_t>(x_indices_shape.size());
  const int64_t x_values_rank = static_cast<int64_t>(x_values_shape.size());
  const int64_t x_dense_shape_rank = static_cast<int64_t>(x_dense_shape_shape.size());
  if ((!IsDynamicRank(x_indices_shape) && x_indices_rank != kIndicesRank) || x_values_rank != 1 ||
      x_dense_shape_rank != 1) {
    MS_EXCEPTION(ValueError) << "For SparseTensorToCSRSparseMatrix, input x_indices should be a 2-D tensor"
                             << ", but got " << x_indices_shape.size() << "-D"
                             << ", input x_values should be a 1-D tensor"
                             << ", but got " << x_values_shape.size() << "-D"
                             << ", input x_dense_shape should be a 1-D tensor"
                             << ", but got " << x_dense_shape_shape.size() << "-D";
  }
  if (!is_dynamic) {
    (void)CheckAndConvertUtils::CheckInteger("x_indices.shape[0] and x_values.shape[0]", x_indices_shape[0], kEqual,
                                             x_values_shape[0], prim_name);
    (void)CheckAndConvertUtils::CheckInteger("x_indices.shape[1] and x_dense_shape.shape[0]", x_indices_shape[1],
                                             kEqual, x_dense_shape_shape[0]);
  }
  auto y_dense_shape_shape = input_args[kInputIndex2]->BuildShape();
  abstract::ShapePtr y_dense_shape_shape_list = y_dense_shape_shape->cast<abstract::ShapePtr>();
  auto y_col_indices_shape = input_args[kInputIndex1]->BuildShape();
  abstract::ShapePtr y_col_indices_shape_list = y_col_indices_shape->cast<abstract::ShapePtr>();
  auto y_values_shape = input_args[kInputIndex1]->BuildShape();
  abstract::ShapePtr y_values_shape_list = y_values_shape->cast<abstract::ShapePtr>();
  abstract::ShapePtr y_batch_pointers_shape_list;
  abstract::ShapePtr y_row_pointers_shape_list;

  if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>() &&
      !input_args[kInputIndex2]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex2]->BuildValue()->isa<None>()) {
    auto dense_shape = input_args[kInputIndex2]->cast<abstract::AbstractTensorPtr>();
    auto dense_shape_ptr = dense_shape->BuildValue();
    auto dense_shape_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("x_dense_shape", dense_shape_ptr, prim_name);

    ShapeVector y_batch_pointers_shape;
    ShapeVector y_row_pointers_shape;
    if (rank_x == kBatchRank) {
      y_batch_pointers_shape.push_back(dense_shape_ptr_tensor[0] + 1);
      y_batch_pointers_shape_list = std::make_shared<abstract::Shape>(y_batch_pointers_shape);

      y_row_pointers_shape.push_back(dense_shape_ptr_tensor[1] * dense_shape_ptr_tensor[0] + dense_shape_ptr_tensor[0]);
      y_row_pointers_shape_list = std::make_shared<abstract::Shape>(y_row_pointers_shape);
    } else {
      y_batch_pointers_shape = {2};
      y_batch_pointers_shape_list = std::make_shared<abstract::Shape>(y_batch_pointers_shape);

      y_row_pointers_shape.push_back(dense_shape_ptr_tensor[0] + 1);
      y_row_pointers_shape_list = std::make_shared<abstract::Shape>(y_row_pointers_shape);
    }
  } else {
    ShapeVector y_batch_pointers_shape = {-1};
    ShapeVector y_row_pointers_shape = {-1};
    y_batch_pointers_shape_list = std::make_shared<abstract::Shape>(y_batch_pointers_shape);
    y_row_pointers_shape_list = std::make_shared<abstract::Shape>(y_row_pointers_shape);
  }

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{y_dense_shape_shape_list, y_batch_pointers_shape_list,
                                        y_row_pointers_shape_list, y_col_indices_shape_list, y_values_shape_list});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseTensorToCSRSparseMatrix, BaseOperator);
AbstractBasePtr SparseTensorToCSRSparseMatrixInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SparseTensorToCSRSparseMatrixInferType(primitive, input_args);
  auto infer_shape = SparseTensorToCSRSparseMatrixInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSparseTensorToCSRSparseMatrixInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseTensorToCSRSparseMatrixInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseTensorToCSRSparseMatrixInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseTensorToCSRSparseMatrixInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseTensorToCSRSparseMatrix, prim::kPrimSparseTensorToCSRSparseMatrix,
                                 AGSparseTensorToCSRSparseMatrixInfer, false);
}  // namespace ops
}  // namespace mindspore
