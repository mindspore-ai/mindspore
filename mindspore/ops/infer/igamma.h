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

#ifndef MINDSPORE_CORE_OPS_IGAMMA_H
#define MINDSPORE_CORE_OPS_IGAMMA_H

#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameIgamma = "Igamma";
/// \brief Calculates lower regularized incomplete Gamma function.
/// Refer to Python API @ref mindspore.ops.Igamma for more details.
class OPS_API Igamma : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Igamma);
  /// \brief Constructor.
  Igamma() : BaseOperator(kNameIgamma) { InitIOName({"a", "x"}, {"z"}); }
};

OPS_API abstract::AbstractBasePtr IgammaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimIgammaPtr = std::shared_ptr<Igamma>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_IGAMMA_H
