/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_APPLY_CAME_PART4_H_
#define MINDSPORE_CORE_OPS_APPLY_CAME_PART4_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyCamePart4 = "ApplyCamePart4";

/// \brief . Compute Part 4 of the CAME Optimizer
/// Refer to Python API @ref mindspore.ops.ApplyCamePart4 for more details.
class MIND_API ApplyCamePart4 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ApplyCamePart4);
  /// \brief Constructor.
  ApplyCamePart4() : BaseOperator(kNameApplyCamePart4) {
    InitIOName({"param", "m", "r", "c", "weight_decay", "lr", "beta3", "sum_r", "sum_u_r", "sum_u_c", "sum_u_rc",
                "global_shape"},
               {"param", "r", "c"});
  }
};

MIND_API abstract::AbstractBasePtr ApplyCamePart4Infer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_APPLY_CAME_PART4_H_
