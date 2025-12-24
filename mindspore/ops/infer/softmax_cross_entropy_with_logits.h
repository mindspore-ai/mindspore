/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#define MINDSPORE_CORE_OPS_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSoftmaxCrossEntropyWithLogits = "SoftmaxCrossEntropyWithLogits";
/// \brief Gets the softmax cross-entropy value between logits and labels with one-hot encoding.
/// Refer to Python API @ref mindspore.ops.SoftmaxCrossEntropyWithLogits for more details.
class OPS_API SoftmaxCrossEntropyWithLogits : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SoftmaxCrossEntropyWithLogits);
  /// \brief Constructor.
  SoftmaxCrossEntropyWithLogits() : BaseOperator(kNameSoftmaxCrossEntropyWithLogits) {
    InitIOName({"features", "labels"}, {"loss", "backprop"});
  }
  /// \brief Init.
  void Init() const {}
};
using kPrimSoftmaxCrossEntropyWithLogitsPtr = std::shared_ptr<SoftmaxCrossEntropyWithLogits>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
