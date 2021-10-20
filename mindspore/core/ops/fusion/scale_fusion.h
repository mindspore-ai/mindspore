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

#ifndef MINDSPORE_CORE_OPS_SCALE_FUSION_H_
#define MINDSPORE_CORE_OPS_SCALE_FUSION_H_
#include "ops/scale.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameScaleFusion = "ScaleFusion";
/// \brief ScaleFusion defined Scale operator prototype of lite.
class MS_CORE_API ScaleFusion : public Scale {
 public:
  /// \brief Constructor.
  ScaleFusion() : Scale(kNameScaleFusion) {}

  /// \brief Destructor.
  ~ScaleFusion() = default;

  MS_DECLARE_PARENT(ScaleFusion, Scale);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axis Define the first axis to do this operation.
  /// \param[in] activation_type Define the activation type.
  void Init(const int64_t axis = -1, const ActivationType &activation_type = NO_ACTIVATION);

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

#endif  // MINDSPORE_CORE_OPS_SCALE_FUSION_H_
