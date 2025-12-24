/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_MOE_F_F_N_H_
#define MINDSPORE_CORE_OPS_MOE_F_F_N_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFFN = "FFN";
class OPS_API FFN : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FFN);
  /// \brief Constructor.
  FFN() : BaseOperator(kNameFFN) {
    InitIOName(
      {"x", "weight1", "weight2", "expert_tokens", "bias1", "bias2", "scale", "offset", "deq_scale1", "deq_scale2"},
      {"y"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.operations._inner_ops.MatMul for the inputs.
  void Init(const std::string &activation, int64_t inner_precise = 0);
  /// \brief Set activation.
  void set_activation(const std::string &activation);
  /// \brief Set inner_precise.
  void set_inner_precise(int64_t inner_precise);
  /// \brief Get activation.
  ///
  /// \return activation.
  std::string get_activation() const;
  /// \brief Get inner_precise.
  ///
  /// \return inner_precise.
  int64_t get_inner_precise() const;
};
OPS_API abstract::AbstractBasePtr FFNInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MOE_F_F_N_H_
