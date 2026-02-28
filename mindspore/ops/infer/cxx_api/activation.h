/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_ACTIVATION_H_
#define MINDSPORE_CORE_OPS_ACTIVATION_H_
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameActivation = "Activation";
/// \brief Activation defined Activation operator prototype of lite.
class OPS_API Activation : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Activation);
  /// \brief Constructor.
  Activation() : BaseOperator(kNameActivation) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] alpha Define a size factor.
  /// \param[in] min_val Define a lower bound.
  /// \param[in] max_val Define a upper bound.
  /// \param[in] activation_type Define the activation type.
  /// \param[in] approximate Define a boolean value to decide whether to use an approximate algorithm, only useful for
  ///            GELU.
  void Init(const float alpha = 0.2, const float min_val = -1.0, const float max_val = 1.0,
            const ActivationType &activation_type = NO_ACTIVATION, bool approximate = false);

  /// \brief Method to set alpha attribute.
  ///
  /// \param[in] alpha Define a size factor.
  void set_alpha(const float alpha);

  /// \brief Method to set min_val attribute.
  ///
  /// \param[in] min_val Define a lower bound.
  void set_min_val(const float min_val);

  /// \brief Method to set max_val attribute.
  ///
  /// \param[in] max_val Define a upper bound.
  void set_max_val(const float max_val);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType &activation_type);

  /// \brief Method to get alpha attribute.
  ///
  /// \return alpha attribute.
  float get_alpha() const;

  /// \brief Method to get min_val attribute.
  ///
  /// \return min_val attribute.
  float get_min_val() const;

  /// \brief Method to get max_val attribute.
  ///
  /// \return max_val attribute.
  float get_max_val() const;

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;

  /// \brief Method to set approximate attribute.
  ///
  /// \param[in] approximate Define a boolean value to decide whether to use an approximate algorithm, only useful for
  ///            GELU.
  void set_approximate(bool approximate);

  /// \brief Method to get approximate attribute.
  ///
  /// \return approximate attribute.
  bool get_approximate() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ACTIVATION_H_
