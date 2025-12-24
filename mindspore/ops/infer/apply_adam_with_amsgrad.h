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

#ifndef MINDSPORE_CORE_OPS_APPLY_ADAM_WITH_AMSGRAD_H_
#define MINDSPORE_CORE_OPS_APPLY_ADAM_WITH_AMSGRAD_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyAdamWithAmsgrad = "ApplyAdamWithAmsgrad";
class OPS_API ApplyAdamWithAmsgrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ApplyAdamWithAmsgrad);
  ApplyAdamWithAmsgrad() : BaseOperator(kNameApplyAdamWithAmsgrad) {
    InitIOName({"var", "m", "v", "vhat", "beta1_power", "beta2_power", "lr", "grad"}, {"var", "m", "v", "vhat"});
  }

  void set_beta1(const float beta1);

  void set_beta2(const float beta2);

  void set_epsilon(const float epsilon);

  void set_use_locking(const bool use_locking);

  float get_beta1() const;

  float get_beta2() const;

  float get_epsilon() const;

  bool get_use_locking() const;
};

OPS_API abstract::AbstractBasePtr ApplyAdamWithAmsgradInfer(const abstract::AnalysisEnginePtr &,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimApplyAdamWithAmsgradPtr = std::shared_ptr<ApplyAdamWithAmsgrad>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_APPLY_ADAM_WITH_AMSGRAD_H_
