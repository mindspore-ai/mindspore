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

#ifndef MINDSPORE_CORE_OPS_GRAD_L1_NORMALIZE_GRAD_H_
#define MINDSPORE_CORE_OPS_GRAD_L1_NORMALIZE_GRAD_H_

#include <string>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameL2NormalizeGrad = "L2NormalizeGrad";
/// \brief L2NormalizeGrad operation. Refer to Python API @ref mindspore.ops.L2NormalizeGrad for more details.
class OPS_API L2NormalizeGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(L2NormalizeGrad);
  /// \brief Constructor.
  L2NormalizeGrad() : BaseOperator(kNameL2NormalizeGrad) {
    InitIOName({"logits", "loss_grad", "labels", "weight", "total_weight"}, {"logits_grad"});
  }

  /// \brief Set axis.
  void set_axis(const int64_t axis);

  /// \brief Set epsilon.
  void set_epsilon(const float epsilon);

  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;

  /// \brief Get epsilon.
  ///
  /// \return epsilon.
  float get_epsilon() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRAD_L1_NORMALIZE_GRAD_H_
