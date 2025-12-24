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

#ifndef MINDSPORE_CORE_OPS_ADJUST_HUE_H_
#define MINDSPORE_CORE_OPS_ADJUST_HUE_H_
#include <memory>
#include <string>
#include <vector>
#include "ops/base_operator.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdjustHue = "AdjustHue";
/// \brief Adjust hue of RGB images.
/// Refer to Python API @ref mindspore.ops.AdjustHue for more details.
class OPS_API AdjustHue : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdjustHue);
  AdjustHue() : BaseOperator(kNameAdjustHue) { InitIOName({"images", "delta"}, {"y"}); }
};

OPS_API abstract::AbstractBasePtr AdjustHueInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args);
using PrimAdjustHuePtr = std::shared_ptr<AdjustHue>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADJUST_HUE_H_
