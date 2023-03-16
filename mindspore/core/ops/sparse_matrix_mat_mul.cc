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

#include "ops/sparse_matrix_mat_mul.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void SparseMatrixMatMulCheckShape(const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int64_t> x1_dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  std::vector<int64_t> x1_batch_pointer =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  std::vector<int64_t> x1_row_pointer =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  std::vector<int64_t> x1_col_indices =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  std::vector<int64_t> x1_values =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  std::vector<int64_t> x2_dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];

  std::vector<ShapeVector> check_shapes = {x1_dense_shape, x1_batch_pointer, x2_dense_shape};
  auto is_dynamic = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamic);
  if (!is_dynamic) {
    const int kInputNoBatch = 2;
    const int kInputWithBatch = 3;
    auto x1_dense_shape_size = x1_dense_shape.size();
    if (x1_dense_shape_size == 0) {
      MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1_dense_shape.size() = " << x1_dense_shape_size
                               << ", which is invalid";
    }
    const int64_t rank_x1 = x1_dense_shape[0];
    const int64_t rank_x2 = (SizeToLong)(x2_dense_shape.size());
    if (rank_x1 != rank_x2) {
      MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1_dense_shape.shape[0] and rank of x2_dense must be the "
                                  "same, but got x1_dense_shape.shape[0] = "
                               << rank_x1 << ", and rank of x2_dense = " << rank_x2 << ".";
    }  // check rank
    if (rank_x1 != kInputNoBatch && rank_x1 != kInputWithBatch) {
      MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, rank of x1_dense_shape must be (2,) or (3,), but got "
                               << rank_x1 << ".";
    }
    if (rank_x2 == kInputWithBatch) {
      int64_t x1_batch_num = x1_batch_pointer[0] - 1;
      int64_t x2_batch_num = x2_dense_shape[0];
      if (x1_batch_num != x2_batch_num) {
        MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1_dense_shape[0] and x2_dense.shape[0] must be the "
                                    "same, but got x1_dense_shape[0] = "
                                 << x1_batch_num << ", and x2_dense.shape[0] = " << x2_batch_num << ".";
      }
    }
  }

  const int kOne = 1;
  if (x1_dense_shape.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1_dense_shape should be 1-D, bug got "
                             << x1_dense_shape.size() << "-D.";
  }
  if (x1_batch_pointer.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1_batch_pointers should be 1-D, bug got "
                             << x1_batch_pointer.size() << "-D.";
  }
  if (x1_row_pointer.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1_row_pointers should be 1-D, bug got "
                             << x1_row_pointer.size() << "-D.";
  }
  if (x1_col_indices.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1_col_indices should be 1-D, bug got "
                             << x1_col_indices.size() << "-D.";
  }
  if (x1_values.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1_values should be 1-D, bug got " << x1_values.size()
                             << "-D.";
  }
}

abstract::ShapePtr SparseMatrixMatMulInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  SparseMatrixMatMulCheckShape(input_args);
  std::vector<int64_t> x2_dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];

  auto transpose_x1 = GetValue<bool>(primitive->GetAttr("transpose_x1"));
  auto transpose_x2 = GetValue<bool>(primitive->GetAttr("transpose_x2"));
  auto adjoint_x1 = GetValue<bool>(primitive->GetAttr("adjoint_x1"));
  auto adjoint_x2 = GetValue<bool>(primitive->GetAttr("adjoint_x2"));
  auto transpose_output = GetValue<bool>(primitive->GetAttr("transpose_output"));
  if (adjoint_x1 && transpose_x1) {
    MS_EXCEPTION(ValueError)
      << "For SparseMatrixMatMul, only one of adjoint_x1 and transpose_x1 may be true, but got adjoint_x1 = "
      << adjoint_x1 << ", and transpose_x1 = " << transpose_x1 << ".";
  }
  if (adjoint_x2 && transpose_x2) {
    MS_EXCEPTION(ValueError)
      << "For SparseMatrixMatMul, only one of adjoint_x2 and transpose_x2 may be true, but got adjoint_x2 = "
      << adjoint_x2 << ", and transpose_x2 = " << transpose_x2 << ".";
  }

  int64_t row_x2 = -1, col_x2 = -1;
  if (!IsDynamicRank(x2_dense_shape)) {
    const int64_t rank_x2 = SizeToLong(x2_dense_shape.size());
    const int64_t kNumTwo = 2;
    // row and col of B
    row_x2 = rank_x2 == kNumTwo ? x2_dense_shape[kInputIndex0] : x2_dense_shape[kInputIndex1];
    col_x2 = rank_x2 == kNumTwo ? x2_dense_shape[kInputIndex1] : x2_dense_shape[kInputIndex2];
  }
  col_x2 = (adjoint_x2 || transpose_x2) ? row_x2 : col_x2;

  // row and col of A
  const int kInputWithBatch = 3;
  if (input_args[0]->isa<abstract::AbstractTensor>() && !input_args[0]->BuildValue()->isa<ValueAny>() &&
      !input_args[0]->BuildValue()->isa<None>()) {
    auto dense_shape_value = input_args[0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(dense_shape_value);
    auto dense_shape_value_ptr = dense_shape_value->BuildValue();
    MS_EXCEPTION_IF_NULL(dense_shape_value_ptr);
    auto dense_shape_value_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("x1_dense_shape", dense_shape_value_ptr, primitive->name());

    const int64_t rank_x1 = static_cast<int64_t>(dense_shape_value_ptr_tensor.size());
    auto row_x1 = dense_shape_value_ptr_tensor[rank_x1 - 2];
    auto col_x1 = dense_shape_value_ptr_tensor[rank_x1 - 1];

    row_x1 = (adjoint_x1 || transpose_x1) ? col_x1 : row_x1;

    int64_t row_y = row_x1;
    int64_t col_y = col_x2;
    if (transpose_output) {
      int64_t temp = col_y;
      col_y = row_y;
      row_y = temp;
    }

    ShapeVector y_dense_shape{};
    if (rank_x1 == kInputWithBatch) {
      y_dense_shape.push_back(dense_shape_value_ptr_tensor[0]);
    }
    y_dense_shape.push_back(row_y);
    y_dense_shape.push_back(col_y);
    return std::make_shared<abstract::Shape>(y_dense_shape);
  } else {
    ShapeVector dense_shape = {-2};
    return std::make_shared<abstract::Shape>(dense_shape);
  }
}

TypePtr SparseMatrixMatMulInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> index_valid_types = {kInt32, kInt64};
  const std::set<TypePtr> values_valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto x1_dense_type = input_args[kInputIndex0]->BuildType();
  auto x1_batch_type = input_args[kInputIndex1]->BuildType();
  auto x1_row_type = input_args[kInputIndex2]->BuildType();
  auto x1_col_type = input_args[kInputIndex3]->BuildType();
  auto x1_values_type = input_args[kInputIndex4]->BuildType();
  auto x2_values_type = input_args[kInputIndex5]->BuildType();
  std::map<std::string, TypePtr> types_values;
  (void)types_values.emplace("x1_values", x1_values_type);
  (void)types_values.emplace("x2_dense", x2_values_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types_values, values_valid_types, prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1_dense_shape", x1_dense_type);
  (void)types.emplace("x1_batch_pointers", x1_batch_type);
  (void)types.emplace("x1_row_pointers", x1_row_type);
  (void)types.emplace("x1_col_indices", x1_col_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, index_valid_types, prim->name());
  return x1_values_type;
}
}  // namespace

AbstractBasePtr SparseMatrixMatMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto infer_type = SparseMatrixMatMulInferType(primitive, input_args);
  auto infer_shape = SparseMatrixMatMulInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseMatrixMatMul, BaseOperator);

// AG means auto generated
class MIND_API AGSparseMatrixMatMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixMatMulInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixMatMulInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixMatMulInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseMatrixMatMul, prim::kPrimSparseMatrixMatMul, AGSparseMatrixMatMulInfer, false);
}  // namespace ops
}  // namespace mindspore
