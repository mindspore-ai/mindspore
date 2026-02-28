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

#ifndef MINDSPORE_CORE_OPS_PAD_FUSION_H_
#define MINDSPORE_CORE_OPS_PAD_FUSION_H_
#include "mindapi/base/types.h"
#include "infer/pad.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePadFusion = "PadFusion";
/// \brief PadFusion defined Pad operator prototype of lite.
class OPS_API PadFusion : public Pad {
 public:
  MIND_API_BASE_MEMBER(PadFusion);
  /// \brief Constructor.
  PadFusion() : Pad(kNamePadFusion) { InitIOName({"x"}, {"y"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] padding_mode Define the padding mode.
  /// \param[in] constant_value Define the padding value.
  void Init(const PaddingMode &padding_mode, const float constant_value);

  /// \brief Method to set padding_mode attribute.
  ///
  /// \param[in] padding_mode Define the padding mode.
  void set_padding_mode(const PaddingMode &padding_mode);

  /// \brief Method to set constant_value attribute.
  ///
  /// \param[in] constant_value Define the padding value.
  void set_constant_value(const float constant_value);

  /// \brief Method to get padding_mode attribute.
  ///
  /// \return padding mode.
  PaddingMode get_padding_mode() const;

  /// \brief Method to get constant_value attribute.
  ///
  /// \return a constant value.
  float get_constant_value() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PAD_FUSION_H_
