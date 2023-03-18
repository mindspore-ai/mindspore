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

#include "ops/sparse_matrix_sparse_mat_mul.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <algorithm>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
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
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int MAX_LENGTH = 100000;

void SparseMatrixSparseMatMulCheckInteger(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int kOne = 1;

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
  std::vector<int64_t> x2_batch_pointer =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
  std::vector<int64_t> x2_row_pointer =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->BuildShape())[kShape];
  std::vector<int64_t> x2_col_indices =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex8]->BuildShape())[kShape];
  std::vector<int64_t> x2_values =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex9]->BuildShape())[kShape];

  (void)CheckAndConvertUtils::CheckInteger("rank of x1_dense_shape", SizeToLong(x1_dense_shape.size()), kEqual, kOne,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x1_batch_pointer", SizeToLong(x1_batch_pointer.size()), kEqual,
                                           kOne, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x1_row_pointer", SizeToLong(x1_row_pointer.size()), kEqual, kOne,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x1_col_indices", SizeToLong(x1_col_indices.size()), kEqual, kOne,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x1_values", SizeToLong(x1_values.size()), kEqual, kOne, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x2_dense_shape", SizeToLong(x2_dense_shape.size()), kEqual, kOne,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x2_batch_pointer", SizeToLong(x2_batch_pointer.size()), kEqual,
                                           kOne, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x2_row_pointer", SizeToLong(x2_row_pointer.size()), kEqual, kOne,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x2_col_indices", SizeToLong(x2_col_indices.size()), kEqual, kOne,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of x2_values", SizeToLong(x2_values.size()), kEqual, kOne, prim_name);
}

abstract::TupleShapePtr SparseMatrixSparseMatMulInferShape(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int64_t> x1_dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  const int64_t rank_x1 = x1_dense_shape[0];
  std::vector<int64_t> x2_dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  const int64_t rank_x2 = x2_dense_shape[0];
  if (rank_x1 != rank_x2) {
    MS_EXCEPTION(ValueError)
      << "For SparseMatrixSparseMatMul, x1_dense_shape.shape[0] and rank of x2_dense must be the "
         "same, but got x1_dense_shape.shape[0] = "
      << rank_x1 << ", and rank of x2_dense = " << rank_x2 << ".";
  }

  SparseMatrixSparseMatMulCheckInteger(primitive, input_args);

  const int kInputNoBatch = 2;
  const int kInputWithBatch = 3;
  if (rank_x1 != kInputNoBatch && rank_x1 != kInputWithBatch) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixSparseMatMul, rank of x1_dense_shape must be (2,) or (3,), but got "
                             << rank_x1 << ".";
  }

  std::vector<int64_t> x1_batch_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  std::vector<int64_t> x2_batch_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];

  if (x1_batch_shape[0] != x2_batch_shape[0]) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixSparseMatMul, x1_batch_shape[0] and x2_batch_shape[0] must be the "
                                "same, but got x1_batch_shape[0] = "
                             << x1_batch_shape[0] << ", and x2_batch_shape[0] = " << x2_batch_shape[0] << ".";
  }

  ShapeVector dense_shape = {x1_dense_shape[0]};
  ShapeVector batch_shape = {x1_batch_shape[0]};
  abstract::ShapePtr y_dense_shape = std::make_shared<abstract::Shape>(dense_shape);
  abstract::ShapePtr y_batch_shape = std::make_shared<abstract::Shape>(batch_shape);
  abstract::ShapePtr y_row_shape = nullptr;
  abstract::ShapePtr y_col_shape = nullptr;
  abstract::ShapePtr y_values_shape = nullptr;

  ShapeVector col_shape = {abstract::Shape::kShapeDimAny};
  ShapeVector values_shape = {abstract::Shape::kShapeDimAny};
  ShapeVector infer_shape_max = {MAX_LENGTH};
  y_col_shape = std::make_shared<abstract::Shape>(col_shape, infer_shape_max);
  y_values_shape = std::make_shared<abstract::Shape>(values_shape, infer_shape_max);

  if (input_args[0]->isa<abstract::AbstractTensor>() && !input_args[0]->BuildValue()->isa<ValueAny>() &&
      !input_args[0]->BuildValue()->isa<None>()) {
    auto dense_shape_value = input_args[0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(dense_shape_value);
    auto dense_shape_value_ptr = dense_shape_value->BuildValue();
    MS_EXCEPTION_IF_NULL(dense_shape_value_ptr);
    auto dense_shape_value_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("dense_shape", dense_shape_value_ptr, primitive->name());
    auto row_value = static_cast<int64_t>(*(dense_shape_value_ptr_tensor.end() - 2));
    auto col_value = static_cast<int64_t>(*(dense_shape_value_ptr_tensor.end() - 1));

    auto transpose_a = GetValue<bool>(primitive->GetAttr(kTransposeA));
    auto transpose_b = GetValue<bool>(primitive->GetAttr(kTransposeB));
    auto adjoint_a = GetValue<bool>(primitive->GetAttr("adjoint_a"));
    auto adjoint_b = GetValue<bool>(primitive->GetAttr("adjoint_b"));

    if (adjoint_a && transpose_a) {
      MS_EXCEPTION(ValueError)
        << "For SparseMatrixSparseMatMul, only one of adjoint_a and transpose_a may be true, but got adjoint_a="
        << adjoint_a << " and transpose_a=" << transpose_a << ".";
    }
    if (adjoint_b && transpose_b) {
      MS_EXCEPTION(ValueError)
        << "For SparseMatrixSparseMatMul, only one of adjoint_b and transpose_b  may be true, but got adjoint_b="
        << adjoint_b << " and transpose_b=" << transpose_b << ".";
    }
    if (adjoint_a || transpose_a) {
      row_value = col_value;
    }

    ShapeVector row_shape = {(x1_batch_shape[0] - 1) * (row_value + 1)};
    y_row_shape = std::make_shared<abstract::Shape>(row_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{y_dense_shape, y_batch_shape, y_row_shape, y_col_shape, y_values_shape});
  } else {
    ShapeVector row_shape = {-1};
    y_row_shape = std::make_shared<abstract::Shape>(row_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{y_dense_shape, y_batch_shape, y_row_shape, y_col_shape, y_values_shape});
  }
}

TuplePtr SparseMatrixSparseMatMulInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> index_valid_types = {kInt32, kInt64};
  const std::set<TypePtr> values_valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto x1_dense_type = input_args[kInputIndex0]->BuildType();
  auto x1_batch_type = input_args[kInputIndex1]->BuildType();
  auto x1_row_type = input_args[kInputIndex2]->BuildType();
  auto x1_col_type = input_args[kInputIndex3]->BuildType();
  auto x1_values_type = input_args[kInputIndex4]->BuildType();

  auto x2_dense_type = input_args[kInputIndex5]->BuildType();
  auto x2_batch_type = input_args[kInputIndex6]->BuildType();
  auto x2_row_type = input_args[kInputIndex7]->BuildType();
  auto x2_col_type = input_args[kInputIndex8]->BuildType();

  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1_dense_shape", x1_dense_type);
  (void)types.emplace("x1_batch_pointers", x1_batch_type);
  (void)types.emplace("x1_row_pointers", x1_row_type);
  (void)types.emplace("x1_col_indices", x1_col_type);
  (void)types.emplace("x2_dense_shape", x2_dense_type);
  (void)types.emplace("x2_batch_pointers", x2_batch_type);
  (void)types.emplace("x2_row_pointers", x2_row_type);
  (void)types.emplace("x2_col_indices", x2_col_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, index_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1_values", x1_values_type, values_valid_types, prim->name());

  return std::make_shared<Tuple>(
    std::vector<TypePtr>{x1_dense_type, x1_batch_type, x1_row_type, x1_col_type, x1_values_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseMatrixSparseMatMul, BaseOperator);
AbstractBasePtr SparseMatrixSparseMatMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  const int64_t input_num = 10;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  auto infer_type = SparseMatrixSparseMatMulInferType(primitive, input_args);
  auto infer_shape = SparseMatrixSparseMatMulInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
// AG means auto generated
class MIND_API AGSparseMatrixSparseMatMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixSparseMatMulInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixSparseMatMulInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixSparseMatMulInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseMatrixSparseMatMul, prim::kPrimSparseMatrixSparseMatMul,
                                 AGSparseMatrixSparseMatMulInfer, false);
}  // namespace ops
}  // namespace mindspore
