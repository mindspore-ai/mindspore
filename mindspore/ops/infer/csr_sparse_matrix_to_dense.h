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

#ifndef MINDSPORE_CORE_OPS_CSR_SPARSE_MATRIX_TO_DENSE
#define MINDSPORE_CORE_OPS_CSR_SPARSE_MATRIX_TO_DENSE
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCSRSparseMatrixToDense = "CSRSparseMatrixToDense";
/// \brief Converts a CSR sparse matrix to its dense form.
/// Refer to Python API @ref mindspore.ops.CSRSparseMatrixToDense for more details.
class OPS_API CSRSparseMatrixToDense : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(CSRSparseMatrixToDense);
  /// \brief Constructor.
  CSRSparseMatrixToDense() : BaseOperator(kNameCSRSparseMatrixToDense) {
    InitIOName({"x_dense_shape", "x_batch_pointers", "x_row_pointers", "x_col_indices", "x_values"}, {"y"});
  }
};
OPS_API abstract::AbstractBasePtr CSRSparseMatrixToDenseInfer(const abstract::AnalysisEnginePtr &,
                                                              const PrimitivePtr &primitive,
                                                              const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CSR_SPARSE_MATRIX_TO_DENSE
