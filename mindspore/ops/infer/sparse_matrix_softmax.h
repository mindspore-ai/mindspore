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

#ifndef MINDSPORE_CORE_OPS_SPARSE_MATRIX_SOFTMAX_H_
#define MINDSPORE_CORE_OPS_SPARSE_MATRIX_SOFTMAX_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseMatrixSoftmax = "SparseMatrixSoftmax";
/// \brief Computes the softmax cross-entropy value between logits and sparse encoding labels.
/// Refer to Python API @ref mindspore.ops.SparseMatrixSoftMax for more details.
class OPS_API SparseMatrixSoftmax : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseMatrixSoftmax);
  /// \brief Constructor.
  SparseMatrixSoftmax() : BaseOperator(kNameSparseMatrixSoftmax) {
    InitIOName({"a_dense_shape", "a_batch_pointers", "a_row_pointer", "a_col_indices", "a_values"},
               {"c_dense_shape", "c_batch_pointers", "c_row_pointer", "c_col_indices", "c_values"});
  }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_MATRIX_SOFTMAX_H_
