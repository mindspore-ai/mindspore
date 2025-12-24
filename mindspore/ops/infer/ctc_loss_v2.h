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

#ifndef MINDSPORE_CORE_OPS_CTC_LOSS_V2_H_
#define MINDSPORE_CORE_OPS_CTC_LOSS_V2_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCTCLossV2 = "CTCLossV2";
/// \brief Calculates the CTC (Connectionist Temporal Classification) loss and the gradient.
/// Refer to Python API @ref mindspore.ops.CTCLossV2 for more details.
class OPS_API CTCLossV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(CTCLossV2);
  /// \brief Constructor.
  CTCLossV2() : BaseOperator(kNameCTCLossV2) {
    InitIOName({"log_probs", "targets", "input_lengths", "target_lengths"}, {"neg_log_likelihood", "log_alpha"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.CTCLossV2 for the inputs.
  void Init() const {}

  /// \brief Get blank.
  ///
  /// \return blank.
  int64_t get_blank() const;

  /// \brief Get reduction.
  ///
  /// \return reduction.
  std::string get_reduction() const;

  /// \brief Get zero_infinity.
  ///
  /// \return zero_infinity.
  bool get_zero_infinity() const;
};

OPS_API abstract::AbstractBasePtr CTCLossV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CTC_LOSS_V2_H_
