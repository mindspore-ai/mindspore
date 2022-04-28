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

#include "ops/sparse_matrix_add.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
namespace {
constexpr auto kDenseShape = "dense_shape";

inline void CheckSparseShape(const size_t sparse_shape_size, const size_t expected_dim, const std::string &arg_name) {
  if (sparse_shape_size != expected_dim) {
    MS_EXCEPTION(mindspore::ValueError) << arg_name << " must be a " << expected_dim
                                        << "-dimensional tensor, but got a " << sparse_shape_size
                                        << "-dimensional tensor.";
  }
}

inline void CheckSparseIndicesDtype(const mindspore::TypePtr dtype, const std::string &arg_name) {
  if (!(dtype->equal(mindspore::kInt16) || dtype->equal(mindspore::kInt32) || dtype->equal(mindspore::kInt64))) {
    MS_EXCEPTION(mindspore::TypeError) << "The dtype of " << arg_name << " must be Int16 or Int32 or Int64, but got "
                                       << dtype->ToString() << ".";
  }
}
}  // namespace
void SparseMatrixAdd::set_dense_shape(const std::vector<int64_t> &shape) {
  (void)this->AddAttr(kDenseShape, api::MakeValue(shape));
}

std::vector<int64_t> SparseMatrixAdd::get_dense_shape() const {
  auto value_ptr = GetAttr(kDenseShape);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void SparseMatrixAdd::Init(const std::vector<int64_t> &csr_a, const std::vector<int64_t> &csr_b) {
  auto op_name = this->name();
  if (csr_a != csr_b) {
    MS_LOG(EXCEPTION) << "For " << op_name << "A shape and B shape should be same, but got A = " << csr_a
                      << ", and B = " << csr_b << ".";
  }
  this->set_dense_shape(csr_a);
}

AbstractBasePtr SparseMatrixAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  const std::string op_name = primitive->name();
  // Addition of two CSR tensors. C = Alpha * A + Beta * B.
  // Eight input (CSR_A(three tensors), CSR_B(three tensors), Alpha, Beta)
  constexpr size_t kSMAInputsNum = 8;
  constexpr size_t kSMAShapeSize = 2;
  constexpr size_t kAIndptrIdx = 0;
  constexpr size_t kAIndicesIdx = 1;
  constexpr size_t kAValuesIdx = 2;
  constexpr size_t kBIndptrIdx = 3;
  constexpr size_t kBIndicesIdx = 4;
  constexpr size_t kBValuesIdx = 5;
  constexpr size_t kAlphaIndex = 6;
  constexpr size_t kBetaIndex = 7;
  mindspore::abstract::CheckArgsSize(op_name, input_args, kSMAInputsNum);
  auto a_indptr = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAIndptrIdx);
  auto a_indices = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAIndicesIdx);
  auto a_values = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAValuesIdx);
  auto b_indptr = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kBIndptrIdx);
  auto b_indices = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kBIndicesIdx);
  auto b_values = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kBValuesIdx);
  auto alpha = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAlphaIndex);
  auto beta = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kBetaIndex);
  MS_EXCEPTION_IF_NULL(a_indptr);
  MS_EXCEPTION_IF_NULL(a_indices);
  MS_EXCEPTION_IF_NULL(a_values);
  MS_EXCEPTION_IF_NULL(b_indptr);
  MS_EXCEPTION_IF_NULL(b_indices);
  MS_EXCEPTION_IF_NULL(b_values);
  MS_EXCEPTION_IF_NULL(alpha);
  MS_EXCEPTION_IF_NULL(beta);
  if (!primitive->HasAttr(kDenseShape)) {
    MS_LOG(EXCEPTION) << "Primitive don't have dense_shape attr!";
  }
  auto dense_shape = GetValue<std::vector<int64_t>>(primitive->GetAttr(kDenseShape));
  // 1-D indptr
  CheckSparseShape(a_indptr->shape()->shape().size(), 1, "A indptr");
  CheckSparseShape(b_indptr->shape()->shape().size(), 1, "B indptr");
  auto a_indices_shape = a_indices->shape()->shape();
  auto b_indices_shape = b_indices->shape()->shape();
  // 1-D indices
  CheckSparseShape(a_indices_shape.size(), 1, "A indices");
  CheckSparseShape(b_indices_shape.size(), 1, "B indices");
  auto a_values_shape = a_values->shape()->shape();
  auto b_values_shape = b_values->shape()->shape();
  // 1-D values
  CheckSparseShape(a_values_shape.size(), 1, "A values");
  CheckSparseShape(b_values_shape.size(), 1, "B values");
  // indices' shape == values' shape
  if (a_indices_shape != a_values_shape || b_values_shape != b_indices_shape) {
    MS_EXCEPTION(ValueError) << "Indices and values must have same shape, but get A indices shape " << a_indices_shape
                             << ", A values shape " << a_values_shape << "; B indices shape " << b_indices_shape
                             << ", B values shape " << b_values_shape;
  }
  auto a_type = a_values->element()->BuildType();
  auto b_type = b_values->element()->BuildType();
  // Values in A and B should have the same type.
  if (a_type->type_id() != b_type->type_id()) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", the two input Matirx A and B should have the same dtype. But get A = "
                      << TypeIdToString(a_type->type_id()) << ", B = " << TypeIdToString(b_type->type_id());
  }
  // Indices should be int16, int32 or int64.
  CheckSparseIndicesDtype(a_indices->element()->BuildType(), op_name);
  CheckSparseIndicesDtype(b_indices->element()->BuildType(), op_name);

  if (dense_shape.size() != kSMAShapeSize) {
    MS_EXCEPTION(ValueError) << "Currently only supports 2-dimensional csr tensor, but got shape length = "
                             << dense_shape.size() << ".";
  }
  if (dense_shape[0] + 1 != a_indptr->shape()->shape()[0]) {
    MS_EXCEPTION(ValueError) << "Indptr must have length (dense_shape[0] + 1), but got "
                             << a_indptr->shape()->shape()[0];
  }
  // Make output a csr(indptr, indices, values)
  // Indices maybe dynamic, max_shape = min(a_idx + b_idx, m * n); min_shape = max(a_idx, b_idx)
  int64_t max_indices_shape_ = std::min(a_indices_shape[0] + b_indices_shape[0], dense_shape[0] * dense_shape[1]);
  int64_t min_indices_shape_ = std::max(a_indices_shape[0], b_indices_shape[0]);
  ShapeVector out_shape{-1};
  ShapeVector min_shape{min_indices_shape_};
  ShapeVector max_shape{max_indices_shape_};
  auto out_indices = std::make_shared<AbstractTensor>(
    a_indices->element()->BuildType(), std::make_shared<mindspore::abstract::Shape>(out_shape, min_shape, max_shape));
  auto out_values = std::make_shared<AbstractTensor>(
    a_type, std::make_shared<mindspore::abstract::Shape>(out_shape, min_shape, max_shape));
  auto out_indptr = std::make_shared<AbstractTensor>(a_indptr->element()->BuildType(), a_indptr->shape()->shape());
  AbstractBasePtrList ret = {out_indptr, out_indices, out_values};
  return std::make_shared<AbstractTuple>(ret);
}
MIND_API_OPERATOR_IMPL(SparseMatrixAdd, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseMatrixAdd, prim::kPrimSparseMatrixAdd, SparseMatrixAddInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
