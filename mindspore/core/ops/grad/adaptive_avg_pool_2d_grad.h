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

#ifndef MINDSPORE_CORE_OPS_ADAPTIVE_AVG_POOL_2D_GRAD__H_
#define MINDSPORE_CORE_OPS_ADAPTIVE_AVG_POOL_2D_GRAD__H_
#include <memory>
#include <vector>
#include <string>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdaptiveAvgPool2DGrad = "AdaptiveAvgPool2DGrad";
class MIND_API AdaptiveAvgPool2DGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdaptiveAvgPool2DGrad);
  AdaptiveAvgPool2DGrad() : BaseOperator(kNameAdaptiveAvgPool2DGrad) {
    InitIOName({"input_grad", "orig_input_shape"}, {"output_grad"});
  }
};

MIND_API abstract::AbstractBasePtr AdaptiveAvgPool2DGradInfer(const abstract::AnalysisEnginePtr &,
                                                              const PrimitivePtr &primitive,
                                                              const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimAdaptiveAvgPool2DGradPtr = std::shared_ptr<AdaptiveAvgPool2DGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADAPTIVE_AVG_POOL_2D_GRAD__H_
