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

#ifndef MINDSPORE_CORE_OPS_NLLLOSS_GRAD_H_
#define MINDSPORE_CORE_OPS_NLLLOSS_GRAD_H_

#include <string>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNLLLossGrad = "NLLLossGrad";
/// \brief NLLLossGrad operation. Refer to Python API @ref mindspore.ops.NLLLossGrad for more details.
class MIND_API NLLLossGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NLLLossGrad);
  /// \brief Constructor.
  NLLLossGrad() : BaseOperator(kNameNLLLossGrad) {
    InitIOName({"logits", "loss_grad", "labels", "weight", "total_weight"}, {"logits_grad"});
  }

  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.NLLLossGrad for the inputs.
  void Init(const Reduction &reduction = Reduction::NONE);

  /// \brief Set reduction.
  void set_reduction(const Reduction &reduction);

  /// \brief Get reduction.
  ///
  /// \return reduction.
  Reduction get_reduction() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NLLLOSS_GRAD_H_
