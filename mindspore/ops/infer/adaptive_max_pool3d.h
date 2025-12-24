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

#ifndef MINDSPORE_CORE_OPS_ADAPTIVE_MAX_POOL3D_H_
#define MINDSPORE_CORE_OPS_ADAPTIVE_MAX_POOL3D_H_

#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdaptiveMaxPool3D = "AdaptiveMaxPool3D";
class OPS_API AdaptiveMaxPool3D : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdaptiveMaxPool3D);
  AdaptiveMaxPool3D() : BaseOperator(kNameAdaptiveMaxPool3D) { InitIOName({"x", "output_size"}, {"y", "argmax"}); }
};
OPS_API abstract::AbstractBasePtr AdaptiveMaxPool3DInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<abstract::AbstractBasePtr> &input_args);
using AdaptiveMaxPool3DPtr = std::shared_ptr<AdaptiveMaxPool3D>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADAPTIVE_MAX_POOL3D_H_
