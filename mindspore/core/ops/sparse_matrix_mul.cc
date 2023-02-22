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

#include "ops/sparse_matrix_mul.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
namespace {}  // namespace

AbstractBasePtr SparseMatrixMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr size_t kAShapeIdx = 0;
  constexpr size_t kABatchPointersIdx = 1;
  constexpr size_t kAIndptrIdx = 2;
  constexpr size_t kAIndicesIdx = 3;
  constexpr size_t kAValuesIdx = 4;
  constexpr size_t kBDenseIdx = 5;
  constexpr size_t kSMAInputsNum = 6;

  mindspore::abstract::CheckArgsSize(op_name, input_args, kSMAInputsNum);
  auto a_shape = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAShapeIdx);
  auto a_batch_pointers = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kABatchPointersIdx);
  auto a_indptr = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAIndptrIdx);
  auto a_indices = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAIndicesIdx);
  auto a_values = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAValuesIdx);
  auto b_dense = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kBDenseIdx);
  MS_EXCEPTION_IF_NULL(a_shape);
  MS_EXCEPTION_IF_NULL(a_batch_pointers);
  MS_EXCEPTION_IF_NULL(a_indptr);
  MS_EXCEPTION_IF_NULL(a_indices);
  MS_EXCEPTION_IF_NULL(a_values);
  MS_EXCEPTION_IF_NULL(b_dense);

  auto out_shape = std::make_shared<AbstractTensor>(a_shape->element()->BuildType(), a_shape->shape()->shape());
  auto out_batch_pointers =
    std::make_shared<AbstractTensor>(a_batch_pointers->element()->BuildType(), a_batch_pointers->shape()->shape());
  auto out_indptr = std::make_shared<AbstractTensor>(a_indptr->element()->BuildType(), a_indptr->shape()->shape());
  auto out_indices = std::make_shared<AbstractTensor>(a_indices->element()->BuildType(), a_indices->shape()->shape());
  auto out_values = std::make_shared<AbstractTensor>(a_values->element()->BuildType(), a_values->shape()->shape());
  AbstractBasePtrList ret = {out_shape, out_batch_pointers, out_indptr, out_indices, out_values};
  return std::make_shared<AbstractTuple>(ret);
}
MIND_API_OPERATOR_IMPL(SparseMatrixMul, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseMatrixMul, prim::kPrimSparseMatrixMul, SparseMatrixMulInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
