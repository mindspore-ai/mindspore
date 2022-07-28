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

#ifndef MINDSPORE_CORE_OPS_EDIT_DISTANCE_H_
#define MINDSPORE_CORE_OPS_EDIT_DISTANCE_H_
#include <memory>

#include "ops/base_operator.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace ops {
constexpr auto kNormalize = "normalize";
/// \brief Computes the Levenshtein Edit Distance of two sparse tensors.
/// Refer to Python API @ref mindspore.ops.EditDistance for more details.
class MIND_API EditDistance : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EditDistance);
  /// \brief Constructor.
  EditDistance() : BaseOperator(prim::kEditDistance) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.EditDistance for the inputs.
  void Init(const bool normalize = true);
  /// \brief Set normalize.
  ///
  /// \param[in] normalize Define whether the output need to be normalized.
  void set_normalize(const bool normalize);
  /// \brief Get normalize.
  ///
  /// \return normalize.
  bool normalize() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EDIT_DISTANCE_H_
