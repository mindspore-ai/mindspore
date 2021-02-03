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

#ifndef MINDSPORE_CORE_OPS_APPLY_MOMENTUM_H_
#define MINDSPORE_CORE_OPS_APPLY_MOMENTUM_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyMomentum = "ApplyMomentum";
class ApplyMomentum : public PrimitiveC {
 public:
  ApplyMomentum() : PrimitiveC(kNameApplyMomentum) {
    InitIOName({"variable", "accumulation", "learning_rate", "gradient", "momentum"}, {"output"});
  }
  ~ApplyMomentum() = default;
  MS_DECLARE_PARENT(ApplyMomentum, PrimitiveC);
  void Init(const bool use_nesterov = false, const bool use_locking = false, const float gradient_scale = 1.0);
  void set_use_nesterov(const bool use_nesterov);
  void set_use_locking(const bool use_locking);
  void set_gradient_scale(const float gradient_scale);
  bool get_use_nesterov() const;
  bool get_use_locking() const;
  float get_gradient_scale() const;
};
AbstractBasePtr ApplyMomentumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args);
using PrimApplyMomentumPtr = std::shared_ptr<ApplyMomentum>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_APPLY_MOMENTUM_H_
