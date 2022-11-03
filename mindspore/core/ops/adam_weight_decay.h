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
#ifndef MINDSPORE_CORE_OPS_ADAM_WEIGHT_DECAY_H_
#define MINDSPORE_CORE_OPS_ADAM_WEIGHT_DECAY_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace ops {
constexpr auto kAdamWeightDecay = "AdamWeightDecay";

class MIND_API AdamWeightDecay : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdamWeightDecay);
  AdamWeightDecay() : BaseOperator(kAdamWeightDecay) {
    InitIOName({"vat", "m", "v", "lr", "beta1", "beta2", "epsilon", "decay", "gradient"}, {"var", "m", "v"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.AdamWeightDecay for the inputs.
  void Init(const bool use_locking = false);
  /// \brief Set use_locking.
  void set_use_locking(const bool use_locking);
  /// \brief Get use_locking.
  ///
  /// \return use_locking.
  bool get_use_locking() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADAM_WEIGHT_DECAY_H_
