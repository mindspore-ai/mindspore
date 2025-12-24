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

#ifndef MINDSPORE_CORE_OPS_SPARSE_TO_DENSE_V2_H_
#define MINDSPORE_CORE_OPS_SPARSE_TO_DENSE_V2_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseToDenseV2 = "SparseToDenseV2";
/// \brief Converts a sparse representation into a dense tensor.
/// Refer to Python API @ref mindspore.ops.SparseToDense for more details.
class OPS_API SparseToDenseV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseToDenseV2);
  /// \brief Constructor.
  SparseToDenseV2() : BaseOperator(kNameSparseToDenseV2) {
    InitIOName({"indices", "output_shape", "values", "default_value"}, {"output"});
  }
  /// \brief Init.
  void Init(const bool validate_indices = true);
  /// \brief Set validate_indices.
  void set_validate_indices(const bool validate_indices);
  /// \brief Get validate_indices.
  ///
  /// \return validate_indices.
  bool get_validate_indices() const;
};
OPS_API abstract::AbstractBasePtr SparseToDenseV2Infer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_TO_DENSE_V2_H_
