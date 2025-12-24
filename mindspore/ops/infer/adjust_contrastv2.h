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

#ifndef MINDSPORE_CORE_OPS_ADJUST_CONTRASTV2_H_
#define MINDSPORE_CORE_OPS_ADJUST_CONTRASTV2_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdjustContrastv2 = "AdjustContrastv2";
/// \brief Adjust the contrast of one or more images.
/// Refer to Python API @ref mindspore.ops.AdjustContrastv2 for more details.
class OPS_API AdjustContrastv2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdjustContrastv2);
  /// \brief Constructor.
  AdjustContrastv2() : BaseOperator(kNameAdjustContrastv2) { InitIOName({"images", "contrast_factor"}, {"y"}); }
};

OPS_API abstract::AbstractBasePtr AdjustContrastv2Infer(const abstract::AnalysisEnginePtr &,
                                                        const PrimitivePtr &primitive,
                                                        const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimAdjustContrastv2Ptr = std::shared_ptr<AdjustContrastv2>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADJUST_CONTRASTV2_H_
