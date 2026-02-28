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

#ifndef MINDSPORE_CORE_OPS_L2_NORMALIZE_FUSION_H_
#define MINDSPORE_CORE_OPS_L2_NORMALIZE_FUSION_H_
#include <vector>

#include "mindapi/base/types.h"
#include "infer/l2_normalize.h"

namespace mindspore {
namespace ops {
constexpr auto kNameL2NormalizeFusion = "L2NormalizeFusion";
/// \brief L2NormalizeFusion defined L2Normalize operator prototype of lite.
class OPS_API L2NormalizeFusion : public L2Normalize {
 public:
  MIND_API_BASE_MEMBER(L2NormalizeFusion);
  /// \brief Constructor.
  L2NormalizeFusion() : L2Normalize(kNameL2NormalizeFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axis Define a axis that the normalization is done along with.
  /// \param[in] epsilon Define a value added to the denominator for numerical stability.
  /// \param[in] activation_type Define the activation type.
  void Init(const std::vector<int64_t> &axis, const float epsilon = 1e-4,
            const ActivationType &activation_type = NO_ACTIVATION);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType &activation_type);

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_L2_NORMALIZE_FUSION_H_
