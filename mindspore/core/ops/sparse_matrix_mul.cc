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

#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sparse_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "ops/sparse_matrix_mul.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
namespace {
abstract::TupleShapePtr SparseMatrixMulInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto a_shape_shape = input_args[kInputIndex0]->BuildShape();
  auto a_batch_pointers_shape = input_args[kInputIndex1]->BuildShape();
  auto a_indptr_shape = input_args[kInputIndex2]->BuildShape();
  auto a_indices_shape = input_args[kInputIndex3]->BuildShape();
  auto a_values_shape = input_args[kInputIndex4]->BuildShape();

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    a_shape_shape, a_batch_pointers_shape, a_indptr_shape, a_indices_shape, a_values_shape});
}

TuplePtr SparseMatrixMulInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();

  constexpr size_t kSMAInputsNum = 6;
  mindspore::abstract::CheckArgsSize(op_name, input_args, kSMAInputsNum);
  auto a_shape_type = input_args[kInputIndex0]->BuildType();
  auto a_batch_pointers_type = input_args[kInputIndex1]->BuildType();
  auto a_indptr_type = input_args[kInputIndex2]->BuildType();
  auto a_indices_type = input_args[kInputIndex3]->BuildType();
  auto a_values_type = input_args[kInputIndex4]->BuildType();

  return std::make_shared<Tuple>(
    std::vector<TypePtr>{a_shape_type, a_batch_pointers_type, a_indptr_type, a_indices_type, a_values_type});
}

AbstractBasePtr SparseMatrixMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SparseMatrixMulInferShape(primitive, input_args),
                                SparseMatrixMulInferType(primitive, input_args));
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseMatrixMul, BaseOperator);
class MIND_API AGSparseMatrixMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixMulInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixMulInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseMatrixMulInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseMatrixMul, prim::kPrimSparseMatrixMul, AGSparseMatrixMulInfer, false);
}  // namespace ops
}  // namespace mindspore
