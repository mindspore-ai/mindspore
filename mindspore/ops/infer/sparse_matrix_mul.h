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

#ifndef MINDSPORE_CORE_OPS_SPARSE_MATRIX_MUL_H_
#define MINDSPORE_CORE_OPS_SPARSE_MATRIX_MUL_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseMatrixMul = "SparseMatrixMul";
/// \brief Computes the elementwise multiply between csr sparse and dense matrix.
/// Refer to Python API @ref mindspore.ops.SparseMatrixMul for
/// more details.
class OPS_API SparseMatrixMul : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseMatrixMul);
  /// \brief Constructor.
  SparseMatrixMul() : BaseOperator(kNameSparseMatrixMul) {
    InitIOName({"a_shape", "a_batch_pointers", "a_indptr", "a_indices", "a_values", "b_dense"},
               {"c_shape", "c_batch_pointers", "c_indptr", "c_indices", "c_values"});
  }
  /// \brief Init.
  /// Refer to the parameters of python API @ref
  /// mindspore.ops._csr_ops.SparseMatrixMul for the inputs.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_MATRIX_MUL_H_
