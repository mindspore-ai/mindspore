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

#ifndef MINDSPORE_CORE_OPS_GATHER_ND_H_
#define MINDSPORE_CORE_OPS_GATHER_ND_H_
#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGatherNd = "GatherNd";
/// \brief Gathers slices from a tensor by indices. Refer to Python API @ref mindspore.ops.GatherNd for more details.
class MIND_API GatherNd : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GatherNd);
  /// \brief Constructor.
  GatherNd() : BaseOperator(kNameGatherNd) { InitIOName({"x1", "x2"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.GatherNd for the inputs.
  void Init() const {}
};
abstract::AbstractBasePtr GatherNdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimGatherNdPtr = std::shared_ptr<GatherNd>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GATHER_ND_H_
