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

#ifndef MINDSPORE_CORE_OPS_PRELU_FUSION_H_
#define MINDSPORE_CORE_OPS_PRELU_FUSION_H_
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePReLUFusion = "PReLUFusion";
/// \brief PReLUFusion defined PReLU operator prototype of lite.
class OPS_API PReLUFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PReLUFusion);
  /// \brief Constructor.
  PReLUFusion() : BaseOperator(kNamePReLUFusion) { InitIOName({"x", "weight"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] channel_shared Define a boolean value to indicate whether channel is shared or not.
  /// \param[in] slope Define a size factor applied to the elements less than 0.
  void Init(const bool channel_shared, const std::vector<float> &slope);

  /// \brief Method to set channel_shared attribute.
  ///
  /// \param[in] channel_shared Define a boolean value to indicate whether channel is shared or not.
  void set_channel_shared(const bool channel_shared);

  /// \brief Method to set slope attribute.
  ///
  /// \param[in] slope Define size factors applied to the elements less than 0.
  void set_slope(const std::vector<float> &slope);

  /// \brief Method to get channel_shared attribute.
  ///
  /// \return a boolean value.
  bool get_channel_shared() const;

  /// \brief Method to get slope attribute.
  ///
  /// \return size factors.
  std::vector<float> get_slope() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PRELU_FUSION_H_
