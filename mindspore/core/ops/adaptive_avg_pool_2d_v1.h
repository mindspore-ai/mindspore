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

#ifndef MINDSPORE_CORE_OPS_ADAPTIVE_AVG_POOL_2D_V1_H_
#define MINDSPORE_CORE_OPS_ADAPTIVE_AVG_POOL_2D_V1_H_
#include <memory>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdaptiveAvgPool2DV1 = "AdaptiveAvgPool2DV1";
class MIND_API AdaptiveAvgPool2DV1 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdaptiveAvgPool2DV1);
  AdaptiveAvgPool2DV1() : BaseOperator(kNameAdaptiveAvgPool2DV1) { InitIOName({"x"}, {"y"}); }
};
abstract::AbstractBasePtr AdaptiveAvgPool2DV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimAdaptiveAvgPool2DV1Ptr = std::shared_ptr<AdaptiveAvgPool2DV1>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADAPTIVE_AVG_POOL_2D_V1_H_
