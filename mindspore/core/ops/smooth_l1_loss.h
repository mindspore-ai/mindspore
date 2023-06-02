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

#ifndef MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_H_
#define MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSmoothL1Loss = "SmoothL1Loss";
/// \brief Computes smooth L1 loss, a robust L1 loss.
/// Refer to Python API @ref mindspore.ops.SmoothL1Loss for more details.
class MIND_API SmoothL1Loss : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SmoothL1Loss);
  /// \brief Constructor.
  SmoothL1Loss() : BaseOperator(kNameSmoothL1Loss) { InitIOName({"prediction", "target"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.SmoothL1Loss for the inputs.
  void Init(const float beta, const std::string reduction = "none");
  /// \brief Set beta.
  void set_beta(const float beta);
  /// \brief Get beta.
  ///
  /// \return beta.
  float get_beta() const;
  /// \brief Set reduction.
  void set_reduction(const std::string reduction);
  /// \brief Get reduction.
  ///
  /// \return reduction.
  std::string get_reduction() const;
};
MIND_API abstract::AbstractBasePtr SmoothL1LossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimSmoothL1LossPtr = std::shared_ptr<SmoothL1Loss>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SMOOTH_L1_LOSS_H_
