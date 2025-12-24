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

#ifndef MINDSPORE_CORE_OPS_BIAS_DROPOUT_ADD_H_
#define MINDSPORE_CORE_OPS_BIAS_DROPOUT_ADD_H_
#include <memory>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBiasDropoutAdd = "BiasDropoutAdd";
/// \brief During training, randomly zeroes some of the elements of the input tensor with probability 1-keep_prob
//// from a Bernoulli distribution. Refer to Python API @ref mindspore.ops.Dropout for more details.
class OPS_API BiasDropoutAdd : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BiasDropoutAdd);
  /// \brief Constructor.
  BiasDropoutAdd() : BaseOperator(kNameBiasDropoutAdd) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.BiasDropoutAdd for the inputs.
  void Init(const float keep_prob = 0.5);
  /// \brief Set keep_prob.
  void set_keep_prob(const float keep_prob);
  /// \brief Get keep_prob.
  ///
  /// \return keep_prob.
  float get_keep_prob() const;
  /// \brief Set seed0.
  void set_seed0(const int64_t seed0);
  /// \brief Get seed0.
  ///
  /// \return seed0.
  int64_t get_seed0() const;
  /// \brief Set seed1.
  void set_seed1(const int64_t seed1);
  /// \brief Get seed1.
  ///
  /// \return seed1.
  int64_t get_seed1() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_BIAS_DROPOUT_ADD_H_
