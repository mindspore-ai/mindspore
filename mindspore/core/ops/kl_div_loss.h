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

#ifndef MINDSPORE_CORE_OPS_KL_DIV_LOSS_H
#define MINDSPORE_CORE_OPS_KL_DIV_LOSS_H

#include <memory>
#include <string>
#include <vector>
#include "ops/base_operator.h"
#include "ops/op_name.h"

namespace mindspore {
namespace ops {
constexpr auto kNameKLDivLoss = "KLDivLoss";
/// \brief Returns the singular value decompositions of one or more matrices.
/// Refer to Python API @ref mindspore.ops.KLDivLoss for more details.
class MIND_API KLDivLoss : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(KLDivLoss);
  /// \brief Constructor.
  KLDivLoss() : BaseOperator(kNameKLDivLoss) { InitIOName({"x", "target"}, {"y"}); }
  explicit KLDivLoss(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x", "target"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.KLDivLoss for the inputs.
  void Init(const std::string &reduction = kMean);
  /// \brief Set reduction.
  void set_reduction(const std::string &reduction);
  /// \brief Get reduction.
  ///
  /// \return reduction.
  std::string get_reduction() const;
};

MIND_API abstract::AbstractBasePtr KLDivLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_KL_DIV_LOSS_H
