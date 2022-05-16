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

#ifndef MINDSPORE_CORE_OPS_SPARSE_ADD_H_
#define MINDSPORE_CORE_OPS_SPARSE_ADD_H_
#include <memory>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseAdd = "SparseAdd";

class MIND_API SparseAdd : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseAdd);
  /// \brief Constructor.
  SparseAdd() : BaseOperator(kNameSparseAdd) {
    InitIOName(
      {
        "a_indices",
        "a_values",
        "b_indices",
        "b_values",
      },
      {"sum_indices", "sum_values"});
  }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._csr_ops.SparseAdd for the inputs.
  void Init(const std::vector<int64_t> &a_shape, const std::vector<int64_t> &b_shape, const float &thresh);
  /// \brief Set dense shape.
  void set_a_dense_shape(const std::vector<int64_t> &shape);
  void set_b_dense_shape(const std::vector<int64_t> &shape);
  void set_thresh(const float &thresh);
  /// \brief Get dense shape.
  ///
  /// \return dense shape.
  std::vector<int64_t> get_a_dense_shape() const;
  std::vector<int64_t> get_b_dense_shape() const;
  float get_thresh() const;
};
abstract::AbstractBasePtr SparseAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_MATRIX_ADD_H_
