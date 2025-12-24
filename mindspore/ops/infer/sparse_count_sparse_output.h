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

#ifndef MINDSPORE_CORE_OPS_SPARSE_COUNT_SPARSE_OUTPUT_H_
#define MINDSPORE_CORE_OPS_SPARSE_COUNT_SPARSE_OUTPUT_H_
#include <memory>
#include "ops_utils/op_constants.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseCountSparseOutput = "SparseCountSparseOutput";
/// \brief Performs sparse-output bin counting for a sparse tensor input.
/// Refer to Python API @ref mindspore.ops.SparseCountSparseOutput for more details.
class OPS_API SparseCountSparseOutput : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseCountSparseOutput);
  /// \brief Constructor.
  SparseCountSparseOutput() : BaseOperator(kNameSparseCountSparseOutput) {
    InitIOName({"indices", "values", "dense_shape", "weights"},
               {"output_indices", "output_values", "output_dense_shape"});
  }
  /// \brief Init.
  void Init(bool binary_output = false, int64_t minlength = -1, int64_t maxlength = -1);

  void set_binary_output(const bool binary_output);

  bool get_binary_output() const;

  void set_minlength(const int64_t &minlength);

  int64_t get_minlength() const;

  void set_maxlength(const int64_t &maxlength);

  int64_t get_maxlength() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_COUNT_SPARSE_OUTPUT_H_
