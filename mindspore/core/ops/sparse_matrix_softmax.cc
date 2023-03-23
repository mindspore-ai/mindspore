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

#include <vector>
#include <string>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "mindapi/src/helper.h"

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
}  // namespace

AbstractBasePtr SparseMatrixSoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  constexpr size_t kInputNum = 5;
  constexpr size_t _dense_shape = 0;
  constexpr size_t _batch_pointers = 1;
  constexpr size_t _row_pointers = 2;
  constexpr size_t _col_indices = 3;
  constexpr size_t _values = 4;
  mindspore::abstract::CheckArgsSize(op_name, input_args, kInputNum);
  auto in_dense_shape = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, _dense_shape);
  auto in_batch_pointers = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, _batch_pointers);
  auto in_row_pointers = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, _row_pointers);
  auto in_col_indices = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, _col_indices);
  auto in_values = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, _values);
  MS_EXCEPTION_IF_NULL(in_dense_shape);
  MS_EXCEPTION_IF_NULL(in_batch_pointers);
  MS_EXCEPTION_IF_NULL(in_row_pointers);
  MS_EXCEPTION_IF_NULL(in_col_indices);
  MS_EXCEPTION_IF_NULL(in_values);

  // 1-D indptr
  CheckSparseMartrixShape(in_row_pointers->shape()->shape().size(), 1, " indptr");

  auto in_col_indices_shape = in_col_indices->shape()->shape();
  // 1-D indices
  CheckSparseMartrixShape(in_col_indices_shape.size(), 1, "A indices");

  auto in_values_shape = in_values->shape()->shape();

  // 1-D values
  CheckSparseMartrixShape(in_values_shape.size(), 1, "A values");

  ShapeVector out_shape{in_col_indices_shape[0]};
  auto out_dense_shape =
    std::make_shared<AbstractTensor>(in_dense_shape->element()->BuildType(), in_dense_shape->shape()->shape());
  auto out_batch_pointers =
    std::make_shared<AbstractTensor>(in_batch_pointers->element()->BuildType(), in_batch_pointers->shape()->shape());
  auto out_col_indices =
    std::make_shared<AbstractTensor>(in_col_indices->element()->BuildType(), in_col_indices->shape()->shape());
  auto out_values = std::make_shared<AbstractTensor>(in_values->element()->BuildType(), in_values->shape()->shape());
  auto out_row_pointers =
    std::make_shared<AbstractTensor>(in_row_pointers->element()->BuildType(), in_row_pointers->shape()->shape());

  AbstractBasePtrList ret = {out_dense_shape, out_batch_pointers, out_row_pointers, out_col_indices, out_values};
  return std::make_shared<AbstractTuple>(ret);
}
MIND_API_OPERATOR_IMPL(SparseMatrixSoftmax, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseMatrixSoftmax, prim::kPrimSparseMatrixSoftmax, SparseMatrixSoftmaxInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
