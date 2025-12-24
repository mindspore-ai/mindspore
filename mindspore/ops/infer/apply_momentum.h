/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameApplyMomentum = "ApplyMomentum";
/// \brief Optimizer that implements the Momentum algorithm.
/// Refer to Python API @ref mindspore.ops.ApplyMomentum for more details.
class OPS_API ApplyMomentum : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ApplyMomentum);
  /// \brief Constructor.
  ApplyMomentum() : BaseOperator(kNameApplyMomentum) {
    InitIOName({"var", "accum", "lr", "grad", "momentum"}, {"var", "accum"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ApplyMomentum for the inputs.
  void Init(const bool use_nesterov = false, const bool use_locking = false, const float gradient_scale = 1.0);
  /// \brief Set use_nesterov.
  void set_use_nesterov(const bool use_nesterov);
  /// \brief Set use_locking.
  void set_use_locking(const bool use_locking);
  /// \brief Set gradient_scale.
  void set_gradient_scale(const float gradient_scale);
  /// \brief Get use_nesterov.
  ///
  /// \return use_nesterov.
  bool get_use_nesterov() const;
  /// \brief Get use_locking.
  ///
  /// \return use_locking.
  bool get_use_locking() const;
  /// \brief Get gradient_scale.
  ///
  /// \return gradient_scale.
  float get_gradient_scale() const;
};
OPS_API abstract::AbstractBasePtr ApplyMomentumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimApplyMomentumPtr = std::shared_ptr<ApplyMomentum>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_APPLY_MOMENTUM_H_
