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

#ifndef MINDSPORE_CORE_OPS_APPLY_CAME_PART3_H_
#define MINDSPORE_CORE_OPS_APPLY_CAME_PART3_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyCamePart3 = "ApplyCamePart3";

/// \brief . Compute Part 3 of the CAME Optimizer
/// Refer to Python API @ref mindspore.ops.ApplyCamePart3 for more details.
class MIND_API ApplyCamePart3 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ApplyCamePart3);
  /// \brief Constructor.
  ApplyCamePart3() : BaseOperator(kNameApplyCamePart3) {
    InitIOName({"u", "m", "eps", "beta1", "clip_threshold", "sum_square_u", "global_shape", "use_first_moment"},
               {"m", "sum_u_r", "sum_u_c", "sum_u_rc"});
  }
};

MIND_API abstract::AbstractBasePtr ApplyCamePart3Infer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_APPLY_CAME_PART3_H_
