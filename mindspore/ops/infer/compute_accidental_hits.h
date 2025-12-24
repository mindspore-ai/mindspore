/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_COMPUTER_ACCIDENTAL_HITS_H_
#define MINDSPORE_CORE_OPS_COMPUTER_ACCIDENTAL_HITS_H_

#include <vector>
#include <memory>

#include "abstract/abstract_value.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameComputeAccidentalHits = "ComputeAccidentalHits";
/// \brief Compute accidental hits of sampled classes which match target classes.
class OPS_API ComputeAccidentalHits : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ComputeAccidentalHits);
  /// \brief Constructor.
  ComputeAccidentalHits() : BaseOperator(kNameComputeAccidentalHits) {
    InitIOName({"true_classes", "sampled_candidates"}, {"indices", "ids", "weights"});
  }
};

OPS_API AbstractBasePtr ComputeAccidentalHitsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args);

using PrimComputeAccidentalHitsPtr = std::shared_ptr<ComputeAccidentalHits>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_COMPUTER_ACCIDENTAL_HITS_H_
