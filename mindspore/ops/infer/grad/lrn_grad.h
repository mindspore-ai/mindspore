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

#ifndef MINDSPORE_CORE_OPS_LRN_GRAD_H_
#define MINDSPORE_CORE_OPS_LRN_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLRNGrad = "LRNGrad";
/// \brief Local Response Normalization's Grad. Refer to Python API @ref mindspore.ops.LRNGrad for more details.
class OPS_API LRNGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LRNGrad);
  /// \brief Constructor.
  LRNGrad() : BaseOperator(kNameLRNGrad) { InitIOName({"x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.LRNGrad for the inputs.
  void Init(const int64_t depth_radius = 5, const float bias = 1.0, const float alpha = 1.0, const float beta = 0.5);
  /// \brief Set depth_radius.
  void set_depth_radius(const int64_t depth_radius);
  /// \brief Set bias.
  void set_bias(const float bias);
  /// \brief Set alpha.
  void set_alpha(const float alpha);
  /// \brief Set beta.
  void set_beta(const float beta);
  /// \brief Get depth_radius.
  ///
  /// \return depth_radius.
  int64_t get_depth_radius() const;
  /// \brief Get bias.
  ///
  /// \return bias.
  float get_bias() const;
  /// \brief Get alpha.
  ///
  /// \return alpha.
  float get_alpha() const;
  /// \brief Get beta.
  ///
  /// \return beta.
  float get_beta() const;
};
OPS_API abstract::AbstractBasePtr LrnGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_LRN_GRAD_H_
