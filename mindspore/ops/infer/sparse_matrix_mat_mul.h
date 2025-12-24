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

#ifndef MINDSPORE_CORE_OPS_SPARSE_MATRIX_MAT_MUL_H_
#define MINDSPORE_CORE_OPS_SPARSE_MATRIX_MAT_MUL_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseMatrixMatMul = "SparseMatrixMatMul";
/// \brief return a matrix multiplication of a sparse matrix a with a dense matrix b;
/// returns a dense matrix a * b, unless either a or b is transposed or adjointed..
/// Refer to Python API @ref mindspore.ops.SparseMatrixMatMul for more details.
class OPS_API SparseMatrixMatMul : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseMatrixMatMul);
  /// \brief Constructor.
  SparseMatrixMatMul() : BaseOperator(kNameSparseMatrixMatMul) {
    InitIOName({"x1_dense_shape", "x1_batch_pointers", "x1_row_pointers", "x1_col_indices", "x1_values", "x2_dense"},
               {"y_dense"});
  }
  /// \brief Init.
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr SparseMatrixMatMulInfer(const abstract::AnalysisEnginePtr &,
                                                          const PrimitivePtr &primitive,
                                                          const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_MATRIX_MAT_MUL_H_
