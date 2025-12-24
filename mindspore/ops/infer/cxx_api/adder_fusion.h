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

#ifndef MINDSPORE_CORE_OPS_ADDER_FUSION_H_
#define MINDSPORE_CORE_OPS_ADDER_FUSION_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "infer/adder.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAdderFusion = "AdderFusion";
/// \brief AdderFusion defined Adder operator prototype of lite.
class OPS_API AdderFusion : public Adder {
 public:
  MIND_API_BASE_MEMBER(AdderFusion);
  /// \brief Constructor.
  AdderFusion() : Adder(kNameAdderFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] in_channel Define the number of input channel.
  /// \param[in] out_channel Define the number of output channel.
  /// \param[in] kernel_size Define the size of the filter kernel.
  /// \param[in] pad_mode Define the padding method.
  /// \param[in] stride Define the moving size of the filter kernel.
  /// \param[in] pad_list Define the concrete padding value on H and W dimension.
  /// \param[in] dilation Define the coefficient of expansion of the filter kernel.
  /// \param[in] group Define the number of group.
  /// \param[in] format Define the format of input tensor.
  /// \param[in] activation_type Define the activation type.
  void Init(const int64_t in_channel, const int64_t out_channel, const std::vector<int64_t> &kernel_size,
            const PadMode &pad_mode, const std::vector<int64_t> &stride, const std::vector<int64_t> &pad_list,
            const std::vector<int64_t> &dilation, const int64_t group, const Format &format,
            const ActivationType activation_type);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType activation_type);

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ADDER_FUSION_H_
