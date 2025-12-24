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

#ifndef MINDSPORE_CORE_OPS_APPLY_ADAGRAD_V2_H_
#define MINDSPORE_CORE_OPS_APPLY_ADAGRAD_V2_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyAdagradV2 = "ApplyAdagradV2";
/// \brief Updates relevant entries according to the adagradv2 scheme.
/// Refer to Python API @ref mindspore.ops.ApplyAdagradV2 for more details.
class OPS_API ApplyAdagradV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ApplyAdagradV2);
  ApplyAdagradV2() : BaseOperator(kNameApplyAdagradV2) { InitIOName({"var", "accum", "lr", "grad"}, {"var", "accum"}); }
  void Init(float epsilon, bool update_slots = true);
  void set_epsilon(const float epsilon);
  /// \brief Get epsilon.
  ///
  /// \return epsilon.
  float get_epsilon() const;

  /// \brief Set update_slots, A bool where if True, accum will be updated. Default: True.
  void set_update_slots(const bool update_slots);
  /// \brief Get update_slots.
  ///
  /// \return update_slots.
  bool get_update_slots() const;
};

OPS_API abstract::AbstractBasePtr ApplyAdagradV2Infer(const abstract::AnalysisEnginePtr &,
                                                      const PrimitivePtr &primitive,
                                                      const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimApplyAdagradV2Ptr = std::shared_ptr<ApplyAdagradV2>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_APPLY_ADAGRAD_V2_H_
