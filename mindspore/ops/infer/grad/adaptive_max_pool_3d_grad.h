/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_ADAPTIVE_MAX_POOL_3D_GRAD_H_
#define MINDSPORE_CORE_OPS_ADAPTIVE_MAX_POOL_3D_GRAD_H_
#include <memory>
#include <set>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdaptiveMaxPool3DGrad = "AdaptiveMaxPool3DGrad";
class OPS_API AdaptiveMaxPool3DGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdaptiveMaxPool3DGrad);
  AdaptiveMaxPool3DGrad() : BaseOperator(kNameAdaptiveMaxPool3DGrad) {
    InitIOName({"input_grad", "x", "argmax"}, {"output_grad"});
  }
};
OPS_API abstract::AbstractBasePtr AdaptiveMaxPool3DGradInfer(const abstract::AnalysisEnginePtr &,
                                                             const PrimitivePtr &primitive,
                                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using AdaptiveMaxPool3DGradPtr = std::shared_ptr<AdaptiveMaxPool3DGrad>;
}  // namespace ops
}  // namespace mindspore

#endif
