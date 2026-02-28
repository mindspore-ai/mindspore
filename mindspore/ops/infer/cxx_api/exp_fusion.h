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

#ifndef MINDSPORE_CORE_OPS_EXP_FUSION_H_
#define MINDSPORE_CORE_OPS_EXP_FUSION_H_

#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameExpFusion = "ExpFusion";
/// \brief ExpFusion defined Exp operator prototype of lite.
class OPS_API ExpFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ExpFusion);
  /// \brief Constructor.
  ExpFusion() : BaseOperator(kNameExpFusion) { InitIOName({"x"}, {"y"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] base Define a base number. If base is -1, it represents e. In addition to this, base must be larger
  ///            than 0.
  /// \param[in] scale Define a size factor of input.
  /// \param[in] shift Define a bias of input.
  void Init(const float base = -1.0, const float scale = 1.0, const float shift = 0.0);

  /// \brief Method to set base attribute.
  ///
  /// \param[in] base Define a base number. If base is -1, it represents e. In addition to this, base must be larger
  ///            than 0. Default is -1.
  void set_base(const float base);

  /// \brief Method to set scale attribute. Default is 1.0.
  ///
  /// \param[in] scale Define a size factor of input.
  void set_scale(const float scale);

  /// \brief Method to set shift attribute. Default is 0.0.
  ///
  /// \param[in] shift Define a bias of input.
  void set_shift(const float shift);

  /// \brief Method to get base attribute.
  ///
  /// \return base number.
  float get_base() const;

  /// \brief Method to get scale attribute.
  ///
  /// \return a size factor of input.
  float get_scale() const;

  /// \brief Method to get shift attribute.
  ///
  /// \return a bias of input.
  float get_shift() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EXP_FUSION_H_
