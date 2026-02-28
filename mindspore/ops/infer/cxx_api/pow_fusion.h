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

#ifndef MINDSPORE_CORE_OPS_POW_FUSION_H_
#define MINDSPORE_CORE_OPS_POW_FUSION_H_
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePowFusion = "PowFusion";
/// \brief PowFusion defined Pow operator prototype of lite.
class OPS_API PowFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PowFusion);
  /// \brief Constructor.
  PowFusion() : BaseOperator(kNamePowFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] scale Define a size factor applied to input.
  /// \param[in] shift Define a bias applied to input.
  void Init(const float &scale, const float &shift);

  /// \brief Method to set scale attribute. Default is 1.0.
  ///
  /// \param[in] scale Define a size factor applied to input.
  void set_scale(const float &scale);

  /// \brief Method to set shift attribute. Default is 0.0.
  ///
  /// \param[in] shift Define a bias applied to input.
  void set_shift(const float &shift);

  /// \brief Method to get scale attribute.
  ///
  /// \return a size factor.
  float get_scale() const;

  /// \brief Method to get shift attribute.
  ///
  /// \return a bias value.
  float get_shift() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_POW_FUSION_H_
