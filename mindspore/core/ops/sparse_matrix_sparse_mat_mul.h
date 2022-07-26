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

#ifndef MINDSPORE_CORE_OPS_SPARSE_MATRIX_SPARSE_MAT_MUL_H_
#define MINDSPORE_CORE_OPS_SPARSE_MATRIX_SPARSE_MAT_MUL_H_
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ops/op_utils.h"
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseMatrixSparseMatMul = "SparseMatrixSparseMatMul";
/// \brief return a matrix multiplication of a sparse matrix a with a sparse matrix b;
/// returns a sparse matrix a * b, unless either a or b is transposed or adjointed..
/// Refer to Python API @ref mindspore.ops.SparseMatrixSparseMatMul for more details.
class MIND_API SparseMatrixSparseMatMul : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseMatrixSparseMatMul);
  /// \brief Constructor.
  SparseMatrixSparseMatMul() : BaseOperator(kNameSparseMatrixSparseMatMul) {
    InitIOName({"x1_dense_shape", "x1_batch_pointers", "x1_row_pointers", "x1_col_indices", "x1_values",
                "x2_dense_shape", "x2_batch_pointers", "x2_row_pointers", "x2_col_indices", "x2_values"},
               {"y_dense_shape", "y_batch_pointers", "y_row_pointers", "y_col_indices", "y_values"});
  }
  bool get_transpose_a() const {
    auto value_ptr = GetAttr(kTransposeA);
    return GetValue<bool>(value_ptr);
  }
  bool get_transpose_b() const {
    auto value_ptr = GetAttr(kTransposeB);
    return GetValue<bool>(value_ptr);
  }
  bool get_adjoint_a() const {
    auto value_ptr = GetAttr(kAdjointA);
    return GetValue<bool>(value_ptr);
  }
  bool get_adjoint_b() const {
    auto value_ptr = GetAttr(kAdjointB);
    return GetValue<bool>(value_ptr);
  }
  /// \brief Init.
  void Init() const {}
};
abstract::AbstractBasePtr SparseMatrixSparseMatMulInfer(const abstract::AnalysisEnginePtr &,
                                                        const PrimitivePtr &primitive,
                                                        const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimSparseMatrixSparseMatMul = std::shared_ptr<SparseMatrixSparseMatMul>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_MATRIX_SPARSE_MAT_MUL_H_
