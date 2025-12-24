/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_H_
#define MINDSPORE_CORE_OPS_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSigmoidCrossEntropyWithLogits = "SigmoidCrossEntropyWithLogits";
/// \brief Uses the given logits to compute sigmoid cross entropy between the logits and the label.
/// Refer to Python API @ref mindspore.ops.SigmoidCrossEntropyWithLogits for more details.
class OPS_API SigmoidCrossEntropyWithLogits : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SigmoidCrossEntropyWithLogits);
  /// \brief Constructor.
  SigmoidCrossEntropyWithLogits() : BaseOperator(kNameSigmoidCrossEntropyWithLogits) {
    InitIOName({"predict", "target"}, {"loss"});
  }
  /// \brief Init.
  void Init() const {}
};
using kPrimSigmoidCrossEntropyWithLogitsPtr = std::shared_ptr<SigmoidCrossEntropyWithLogits>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_H_
