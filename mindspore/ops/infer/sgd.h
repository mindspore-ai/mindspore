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

#ifndef MINDSPORE_CORE_OPS_SGD_H_
#define MINDSPORE_CORE_OPS_SGD_H_
#include <memory>
#include <vector>

#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSGD = "SGD";
/// \brief Computes the stochastic gradient descent.
/// Refer to Python API @ref mindspore.ops.SGD for more details.
class OPS_API SGD : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SGD);
  /// \brief Constructor.
  SGD() : BaseOperator(kNameSGD) {
    InitIOName({"parameters", "gradient", "learning_rate", "accum", "momentum", "stat"}, {"output"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.SGD for the inputs.
  void Init(const float dampening = 0.0, const float weight_decay = 0.0, const bool nesterov = false);
  /// \brief Set dampening.
  void set_dampening(const float dampening);
  /// \brief Set weight_decay.
  void set_weight_decay(const float weight_decay);
  /// \brief Set nesterov.
  void set_nesterov(const bool nesterov);
  /// \brief Get dampening.
  ///
  /// \return dampening.
  float get_dampening() const;
  /// \brief Get weight_decay.
  ///
  /// \return weight_decay.
  float get_weight_decay() const;
  /// \brief Get nesterov.
  ///
  /// \return nesterov.
  bool get_nesterov() const;
};
OPS_API abstract::AbstractBasePtr SGDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimSGD = std::shared_ptr<SGD>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SGD_H_
