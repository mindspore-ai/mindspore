/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_GRAD_H_
#define MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_GRAD_H_
#include <memory>
#include <vector>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSmoothL1LossGrad = "SmoothL1LossGrad";
class SmoothL1LossGrad : public PrimitiveC {
 public:
  SmoothL1LossGrad() : PrimitiveC(kNameSmoothL1LossGrad) {}
  ~SmoothL1LossGrad() = default;
  MS_DECLARE_PARENT(SmoothL1LossGrad, PrimitiveC);
  void Init();
  void Init(const float beta);
  void set_beta(const float beta);
  float get_beta() const;
};
AbstractBasePtr SmoothL1LossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args);
using PrimSmoothL1LossGradPtr = std::shared_ptr<SmoothL1LossGrad>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_GRAD_H_
