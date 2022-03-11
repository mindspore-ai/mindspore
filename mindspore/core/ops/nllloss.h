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

#ifndef MINDSPORE_CORE_OPS_NLLLOSS_H_
#define MINDSPORE_CORE_OPS_NLLLOSS_H_

#include <string>

#include "ops/primitive_c.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNLLLoss = "NLLLoss";
/// \brief NLLLoss operation. Refer to Python API @ref mindspore.ops.NLLLoss for more details.
class MS_CORE_API NLLLoss : public PrimitiveC {
 public:
  /// \brief Constructor.
  NLLLoss() : PrimitiveC(kNameNLLLoss) { InitIOName({"logits", "labels", "weight"}, {"loss", "total_weight"}); }

  /// \brief Destructor.
  ~NLLLoss() = default;

  MS_DECLARE_PARENT(NLLLoss, PrimitiveC);

  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.NLLLoss for the inputs.
  void Init(const Reduction &reduction = NONE);

  /// \brief Set reduction.
  void set_reduction(const Reduction &reduction);

  /// \brief Get reduction.
  ///
  /// \return reduction.
  Reduction get_reduction() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NLLLOSS_H_
