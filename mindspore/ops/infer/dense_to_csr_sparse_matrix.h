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

#ifndef MINDSPORE_CORE_OPS_DENSE_TO_CSR_SPARSE_MATRIX
#define MINDSPORE_CORE_OPS_DENSE_TO_CSR_SPARSE_MATRIX
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDenseToCSRSparseMatrix = "DenseToCSRSparseMatrix";
/// \brief Converts a dense matrix to its CSR sparse form.
/// Refer to Python API @ref mindspore.ops.DenseToCSRSparseMatrix for more details.
class OPS_API DenseToCSRSparseMatrix : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DenseToCSRSparseMatrix);
  /// \brief Constructor.
  DenseToCSRSparseMatrix() : BaseOperator(kNameDenseToCSRSparseMatrix) {
    InitIOName({"dense_input", "indices"},
               {"y_dense_shape", "y_batch_pointers", "y_row_pointers", "y_col_indices", "y_values"});
  }
};
OPS_API abstract::AbstractBasePtr DenseToCSRSparseMatrixInfer(const abstract::AnalysisEnginePtr &,
                                                              const PrimitivePtr &primitive,
                                                              const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DENSE_TO_CSR_SPARSE_MATRIX
