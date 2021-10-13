/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSigmoidCrossEntropyWithLogits = "SigmoidCrossEntropyWithLogits";
/// \brief Uses the given logits to compute sigmoid cross entropy between the logits and the label.
/// Refer to Python API @ref mindspore.ops.SigmoidCrossEntropyWithLogits for more details.
class MS_CORE_API SigmoidCrossEntropyWithLogits : public PrimitiveC {
 public:
  /// \brief Constructor.
  SigmoidCrossEntropyWithLogits() : PrimitiveC(kNameSigmoidCrossEntropyWithLogits) {
    InitIOName({"predict", "target"}, {"loss"});
  }
  /// \brief Destructor.
  ~SigmoidCrossEntropyWithLogits() = default;
  MS_DECLARE_PARENT(SigmoidCrossEntropyWithLogits, PrimitiveC);
  /// \brief Init.
  void Init() {}
};
AbstractBasePtr SigmoidCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args);
using PrimSigmoidCrossEntropyWithLogitsPtr = std::shared_ptr<SigmoidCrossEntropyWithLogits>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_H_
