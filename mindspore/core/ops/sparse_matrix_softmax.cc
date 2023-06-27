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

#include "ops/sparse_matrix_softmax.h"

#include <memory>
#include <string>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sparse_ops.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
namespace {
inline void CheckSparseMartrixShape(const size_t sparse_shape_size, const size_t expected_dim,
                                    const std::string &arg_name) {
  if (sparse_shape_size != expected_dim) {
    MS_EXCEPTION(mindspore::ValueError) << arg_name << " must be a " << expected_dim
                                        << "-dimensional tensor, but got a " << sparse_shape_size
                                        << "-dimensional tensor.";
  }
}

abstract::TupleShapePtr SparseMatrixSoftmaxInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr size_t kInputNum = 5;
  mindspore::abstract::CheckArgsSize(op_name, input_args, kInputNum);
  auto dense_shape_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto batch_pointers_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto row_pointers_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto col_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];

  // 1-D indptr
  CheckSparseMartrixShape(row_pointers_shape.size(), 1, " indptr");
  // 1-D indices
  CheckSparseMartrixShape(col_indices_shape.size(), 1, "A indices");
  // 1-D values
  CheckSparseMartrixShape(values_shape.size(), 1, "A values");

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    std::make_shared<abstract::Shape>(dense_shape_shape), std::make_shared<abstract::Shape>(batch_pointers_shape),
    std::make_shared<abstract::Shape>(row_pointers_shape), std::make_shared<abstract::Shape>(col_indices_shape),
    std::make_shared<abstract::Shape>(values_shape)});
}

TuplePtr SparseMatrixSoftmaxInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr size_t kInputNum = 5;
  mindspore::abstract::CheckArgsSize(op_name, input_args, kInputNum);
  auto dense_shape_type = input_args[kInputIndex0]->BuildType();
  auto batch_pointers_type = input_args[kInputIndex1]->BuildType();
  auto row_pointers_type = input_args[kInputIndex2]->BuildType();
  auto col_indices_type = input_args[kInputIndex3]->BuildType();
  auto values_type = input_args[kInputIndex4]->BuildType();

  return std::make_shared<Tuple>(
    std::vector<TypePtr>{dense_shape_type, batch_pointers_type, row_pointers_type, col_indices_type, values_type});
}

AbstractBasePtr SparseMatrixSoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SparseMatrixSoftmaxInferShape(primitive, input_args),
                                SparseMatrixSoftmaxInferType(primitive, input_args));
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseMatrixSoftmax, BaseOperator);
class MIND_API AGSparseMatrixSoftmaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixSoftmaxInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixSoftmaxInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixSoftmaxInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseMatrixSoftmax, prim::kPrimSparseMatrixSoftmax, AGSparseMatrixSoftmaxInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
