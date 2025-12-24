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

#ifndef MINDSPORE_CORE_OPS_APPLY_ADAM_WITH_AMSGRAD_V2_H_
#define MINDSPORE_CORE_OPS_APPLY_ADAM_WITH_AMSGRAD_V2_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyAdamWithAmsgradV2 = "ApplyAdamWithAmsgradV2";
class OPS_API ApplyAdamWithAmsgradV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ApplyAdamWithAmsgradV2);
  ApplyAdamWithAmsgradV2() : BaseOperator(kNameApplyAdamWithAmsgradV2) {
    InitIOName({"var", "m", "v", "vhat", "beta1_power", "beta2_power", "lr", "beta1", "beta2", "epsilon", "grad"},
               {"var", "m", "v", "vhat"});
  }

  void set_use_locking(const bool use_locking);

  bool get_use_locking() const;
};

OPS_API abstract::AbstractBasePtr ApplyAdamWithAmsgradV2Infer(const abstract::AnalysisEnginePtr &,
                                                              const PrimitivePtr &primitive,
                                                              const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimApplyAdamWithAmsgradV2Ptr = std::shared_ptr<ApplyAdamWithAmsgradV2>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_APPLY_ADAM_WITH_AMSGRAD_H_
