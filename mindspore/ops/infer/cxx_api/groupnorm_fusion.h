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

#ifndef MINDSPORE_CORE_OPS_FUSION_GROUPNORM_FUSION_H_
#define MINDSPORE_CORE_OPS_FUSION_GROUPNORM_FUSION_H_
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGroupNormFusion = "GroupNormFusion";
/// \brief GroupNormFusion defined GroupNormFusion operator prototype of lite.
class OPS_API GroupNormFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GroupNormFusion);
  /// \brief Constructor.
  GroupNormFusion() : BaseOperator(kNameGroupNormFusion) { InitIOName({"x"}, {"y"}); }
  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] num_groups Define group number.
  /// \param[in] eps Define epsilon.
  void Init(const int64_t num_groups, const float eps = 1e-5, bool affine = true);

  /// \brief Method to set epsilon attribute.
  ///
  /// \param[in] epsilon Define epsilon for numerical stability.
  void set_epsilon(const float epsilon);

  /// \brief Method to set num_groups attribute.
  ///
  /// \param[in] num_groups Define number of groups to separate the channels into.
  void set_num_groups(const int64_t num_groups);

  /// \brief Method to set affine attribute.
  ///
  /// \param[in] affine Define whether this ops has learnable parameters.
  void set_affine(const bool affine);

  /// \brief Method to get epsilon attribute.
  ///
  /// \return epsilon attribute.
  float get_epsilon() const;

  /// \brief Method to get num_groups attribute.
  ///
  /// \return num_groups attribute.
  int64_t get_num_groups() const;

  /// \brief Method to get affine attribute.
  ///
  /// \return affine attribute.
  bool get_affine() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FUSION_GROUPNORM_FUSION_H_
