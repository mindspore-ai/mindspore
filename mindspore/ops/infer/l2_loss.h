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

#ifndef MINDSPORE_CORE_OPS_L2_LOSS_H_
#define MINDSPORE_CORE_OPS_L2_LOSS_H_
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameL2Loss = "L2Loss";
class OPS_API L2Loss : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(L2Loss);
  L2Loss() : BaseOperator(kNameL2Loss) { InitIOName({"x"}, {"output"}); }
  void Init() const {}
};
OPS_API abstract::AbstractBasePtr L2LossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimL2LossPtr = std::shared_ptr<L2Loss>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_L2_LOSS_H_
