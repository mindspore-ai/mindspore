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

#include "ops/sparse_matrix_ordering_amd.h"

#include <memory>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseMatrixOrderingAMDInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto d_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto b_ptrs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto r_ptrs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto c_ind_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  const int64_t kZero = 0, kOne = 1, kDefalutRank = 2, kBatchRank = 3;
  const int64_t rank = d_shape_shape[kZero];
  const int64_t num_batch = b_ptrs_shape[kZero] - 1;
  const int64_t num_rows = r_ptrs_shape[kZero] / num_batch - 1;

  if (d_shape_shape.size() != kOne || c_ind_shape.size() != kOne || values_shape.size() != kOne ||
      r_ptrs_shape.size() != kOne || b_ptrs_shape.size() != kOne) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', each input should be 1-D, but got "
                             << "'x_dense_shape' rank " << d_shape_shape.size() << ", 'x_batch_pointers' rank "
                             << b_ptrs_shape.size() << ", 'x_row_pointers' rank " << r_ptrs_shape.size()
                             << ", 'x_col_indices' rank " << c_ind_shape.size() << ", 'x_values' rank "
                             << values_shape.size() << ".";
  }
  if (rank != kDefalutRank && rank != kBatchRank) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', dense form of the input "
                             << "should have rank 2 or 3, but got " << d_shape_shape[kZero] << ".";
  }
  if (values_shape[kZero] != c_ind_shape[kZero]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'col_indices' and 'values' "
                             << "should have the same length.";
  }

  ShapeVector y_shape;
  if (rank == kBatchRank) {
    y_shape.push_back(num_batch);
  }
  y_shape.push_back(num_rows);
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr SparseMatrixOrderingAMDInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> dense_shape_valid_types = {kInt64};
  const std::set<TypePtr> indices_pointer_valid_types = {kInt32};
  const std::set<TypePtr> values_valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto dense_shape_type = input_args[kInputIndex0]->BuildType();
  auto batch_type = input_args[kInputIndex1]->BuildType();
  auto row_type = input_args[kInputIndex2]->BuildType();
  auto col_type = input_args[kInputIndex3]->BuildType();
  auto value_type = input_args[kInputIndex4]->BuildType();

  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dense_shape", dense_shape_type, dense_shape_valid_types,
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_batch_pointers", batch_type, indices_pointer_valid_types,
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_row_pointers", row_type, indices_pointer_valid_types,
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_col_indices", col_type, indices_pointer_valid_types,
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_values", value_type, values_valid_types, prim->name());
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace

AbstractBasePtr SparseMatrixOrderingAMDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  constexpr int inputs_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, inputs_num, prim->name());
  auto infer_type = SparseMatrixOrderingAMDInferType(prim, input_args);
  auto infer_shape = SparseMatrixOrderingAMDInferShape(prim, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseMatrixOrderingAMD, BaseOperator);

// AG means auto generated
class MIND_API AGSparseMatrixOrderingAMDInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixOrderingAMDInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixOrderingAMDInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixOrderingAMDInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseMatrixOrderingAMD, prim::kPrimSparseMatrixOrderingAMD,
                                 AGSparseMatrixOrderingAMDInfer, false);
}  // namespace ops
}  // namespace mindspore
