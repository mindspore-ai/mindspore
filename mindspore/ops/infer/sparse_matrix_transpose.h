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

#ifndef MINDSPORE_CORE_OPS_SPARSE_MATRIX_TRANSPOSE_H_
#define MINDSPORE_CORE_OPS_SPARSE_MATRIX_TRANSPOSE_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseMatrixTranspose = "SparseMatrixTranspose";
/// \brief Return the transpose of input CSR tensor.
class OPS_API SparseMatrixTranspose : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseMatrixTranspose);
  /// \brief Constructor.
  SparseMatrixTranspose() : BaseOperator(kNameSparseMatrixTranspose) {
    InitIOName({"x_dense_shape", "x_batch_pointers", "x_row_pointers", "x_col_indices", "x_values"},
               {"y_dense_shape", "y_batch_pointers", "y_row_pointers", "y_col_indices", "y_values"});
  }

  void Init(const bool conjugate = false);

  void set_conjugate(const bool conjugate);

  bool get_conjugate() const;
};

OPS_API abstract::AbstractBasePtr SparseMatrixTransposeInfer(const abstract::AnalysisEnginePtr &,
                                                             const PrimitivePtr &primitive,
                                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimSparseMatrixTransposePtr = std::shared_ptr<SparseMatrixTranspose>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_MATRIX_TRANSPOSE_H_
