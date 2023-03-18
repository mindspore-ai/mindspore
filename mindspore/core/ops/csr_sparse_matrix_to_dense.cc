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

#include "ops/csr_sparse_matrix_to_dense.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>

#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "abstract/abstract_value.h"
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
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CSRSparseMatrixToDenseInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kMinusOne = -1;
  const int64_t kZero = 0;
  const int64_t kOne = 1;
  const int64_t kDefalutRank = 2;
  const int64_t kBatchRank = 3;
  CheckInputShapeEmpty(primitive->name(), input_args);
  auto d_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto b_ptrs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto r_ptrs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto c_ind_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  const int64_t rank = d_shape_shape[kZero];
  std::vector<uint64_t> tensor_ranks{d_shape_shape.size(), c_ind_shape.size(), values_shape.size(), r_ptrs_shape.size(),
                                     b_ptrs_shape.size()};
  if (std::any_of(tensor_ranks.cbegin(), tensor_ranks.cend(),
                  [&kOne](const uint64_t i) { return i != static_cast<uint64_t>(kOne); })) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', each input should be 1-D, but got "
                             << "'x_dense_shape' rank " << d_shape_shape.size() << ", 'x_batch_pointers' rank "
                             << b_ptrs_shape.size() << ", 'x_row_pointers' rank " << r_ptrs_shape.size()
                             << ", 'x_col_indices' rank " << c_ind_shape.size() << ", 'x_values' rank "
                             << values_shape.size() << ".";
  }
  // Dynamic Rank
  std::vector<ShapeVector> tensor_shapes{d_shape_shape, c_ind_shape, values_shape, r_ptrs_shape, b_ptrs_shape};
  if (std::any_of(tensor_shapes.cbegin(), tensor_shapes.cend(),
                  [](const ShapeVector shp) { return IsDynamicRank(shp); })) {
    ShapeVector dense_shape = {-2};
    return std::make_shared<abstract::Shape>(dense_shape);
  }
  if (!IsDynamic(d_shape_shape) && rank != kDefalutRank && rank != kBatchRank) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', dense form of the input "
                             << "should have rank 2 or 3, but got " << d_shape_shape[kZero] << ".";
  }
  // Dynamic Shape
  if (input_args[kInputIndex0]->isa<abstract::AbstractTensor>() &&
      (input_args[kInputIndex0]->BuildValue()->isa<ValueAny>() ||
       input_args[kInputIndex0]->BuildValue()->isa<None>())) {
    ShapeVector dense_shape;
    auto shape_size = d_shape_shape[kZero];
    dense_shape.resize(static_cast<size_t>(shape_size), kMinusOne);
    return std::make_shared<abstract::Shape>(dense_shape);
  }
  // Static Shape
  if (!IsDynamic(values_shape) && !IsDynamic(c_ind_shape) && values_shape[kZero] != c_ind_shape[kZero]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'col_indices' and 'values' "
                             << "should have the same length.";
  }
  auto shape_abs_ptr = input_args[kInputIndex0];
  MS_EXCEPTION_IF_NULL(shape_abs_ptr);
  ShapeVector y_shape;
  auto d_shape_value = shape_abs_ptr->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(d_shape_value);
  auto d_shape_value_ptr = d_shape_value->BuildValue();
  MS_EXCEPTION_IF_NULL(d_shape_value_ptr);
  auto d_shape_value_ptr_tensor =
    CheckAndConvertUtils::CheckTensorIntValue("x_dense_shape", d_shape_value_ptr, primitive->name());
  for (int64_t i = kZero; i < rank; i++) {
    if (static_cast<int64_t>(d_shape_value_ptr_tensor[static_cast<size_t>(i)]) <= kZero) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', each element of 'x_dense_shape' must be greater than 0.";
    }
    y_shape.push_back(d_shape_value_ptr_tensor[static_cast<ShapeVector::size_type>(i)]);
  }
  int64_t batch_size = kOne;
  int64_t row_num = d_shape_value_ptr_tensor[kZero];
  if (rank == kBatchRank) {
    batch_size = d_shape_value_ptr_tensor[kZero], row_num = d_shape_value_ptr_tensor[kOne];
  }
  if (!IsDynamic(b_ptrs_shape) && !IsDynamic(r_ptrs_shape) &&
      (b_ptrs_shape[kZero] != (batch_size + kOne) || r_ptrs_shape[kZero] != batch_size * (row_num + kOne))) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', batch size of the input is " << batch_size
                             << ", row numbers of the input is " << row_num << ", so shape of 'x_batch_pointers' "
                             << "should be (" << batch_size + kOne << "), but got (" << b_ptrs_shape[kZero] << ")"
                             << ", shape of 'x_row_pointers' should be (" << batch_size * (row_num + kOne) << "), "
                             << "but got (" << r_ptrs_shape[kZero] << ").";
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr CSRSparseMatrixToDenseInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  const std::set<TypePtr> valid_values_types = {kFloat64, kFloat32, kComplex128, kComplex64};
  const std::set<TypePtr> valid_indices_types = {kInt32, kInt64};
  std::map<std::string, TypePtr> indices_args;
  (void)indices_args.emplace("x_dense_shape", input_args[kInputIndex0]->BuildType());
  (void)indices_args.emplace("x_batch_pointers", input_args[kInputIndex1]->BuildType());
  (void)indices_args.emplace("x_row_pointers", input_args[kInputIndex2]->BuildType());
  (void)indices_args.emplace("x_col_indices", input_args[kInputIndex3]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(indices_args, valid_indices_types, op_name);
  auto values_type = input_args[kInputIndex4]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("x_values", values_type, valid_values_types, op_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(CSRSparseMatrixToDense, BaseOperator);
AbstractBasePtr CSRSparseMatrixToDenseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = CSRSparseMatrixToDenseInferType(primitive, input_args);
  auto shapes = CSRSparseMatrixToDenseInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGCSRSparseMatrixToDenseInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CSRSparseMatrixToDenseInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CSRSparseMatrixToDenseInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CSRSparseMatrixToDenseInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CSRSparseMatrixToDense, prim::kPrimCSRSparseMatrixToDense,
                                 AGCSRSparseMatrixToDenseInfer, false);
}  // namespace ops
}  // namespace mindspore
