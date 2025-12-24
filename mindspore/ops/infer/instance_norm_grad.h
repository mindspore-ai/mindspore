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

#ifndef MINDSPORE_CORE_OPS_INSTANCE_NORM_GRAD_H_
#define MINDSPORE_CORE_OPS_INSTANCE_NORM_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameInstanceNormGrad = "InstanceNormGrad";
/// \brief InstanceNormGrad defined the InstanceNormGrad operator prototype.
class OPS_API InstanceNormGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(InstanceNormGrad);
  /// \brief Constructor.
  InstanceNormGrad() : BaseOperator(kNameInstanceNormGrad) {}

  /// \brief Method to init the op's attributes
  ///
  /// \param[in] epsilon Define a value added to the denominator for numerical stability.
  void Init(const float epsilon = 0.00001);

  /// \brief Method to set epsilon attribute.
  ///
  /// \param[in] epsilon Define a value added to the denominator for numerical stability.
  void set_epsilon(const float epsilon);

  /// \brief Method to get epsilon attribute.
  ///
  /// \return a value.
  float get_epsilon() const;

  /// \brief Method to set inplace_algo attribute.
  ///
  /// \param[in] inplace_algo Define a value added to the denominator for numerical stability.
  void set_inplace_algo(const std::string inplace_algo);

  /// \brief Method to get inplace_algo attribute.
  ///
  /// \return a value.
  std::string get_inplace_algo() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INSTANCE_NORM_GRAD_H_
