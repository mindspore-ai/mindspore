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

#ifndef MINDSPORE_CORE_OPS_CONV2D_BACKPROP_FILTER_FUSION_H_
#define MINDSPORE_CORE_OPS_CONV2D_BACKPROP_FILTER_FUSION_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "infer/grad/conv2d_backprop_filter.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConv2DBackpropFilterFusion = "Conv2DBackpropFilterFusion";
/// \brief Conv2DBackpropFilterFusion defined Conv2DBackpropFilter operator prototype of lite.
class OPS_API Conv2DBackpropFilterFusion : public Conv2DBackpropFilter {
 public:
  MIND_API_BASE_MEMBER(Conv2DBackpropFilterFusion);
  /// \brief Constructor.
  Conv2DBackpropFilterFusion() : Conv2DBackpropFilter(kNameConv2DBackpropFilterFusion) {
    InitIOName({"out_backprop", "input", "filter_sizes"}, {"output"});
  }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] out_channel Define the number of output channel.
  /// \param[in] kernel_size Define the size of the filter kernel.
  /// \param[in] pad_mode Define the padding method.
  /// \param[in] pad_list Define the concrete padding value on H and W dimension.
  /// \param[in] mode Define the category of conv, which is useless on lite.
  /// \param[in] stride Define the moving size of the filter kernel.
  /// \param[in] dilation Define the coefficient of expansion of the filter kernel, which is useful for dilated
  ///            convolution.
  /// \param[in] group Define the number of group.
  /// \param[in] format Define the format of input tensor.
  /// \param[in] activation_type Define the activation type.
  void Init(const int64_t out_channel, const std::vector<int64_t> &kernel_size, const PadMode &pad_mode = VALID,
            const std::vector<int64_t> &pad_list = {0, 0, 0, 0}, const int64_t mode = 1,
            const std::vector<int64_t> &stride = {1, 1}, const std::vector<int64_t> &dilation = {1, 1, 1, 1},
            const int64_t group = 1, const Format &format = NCHW, const ActivationType activation_type = NO_ACTIVATION);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType activation_type);

  /// \brief Method to set in_channel attribute.
  ///
  /// \param[in] in_channel Define the number of input channel.
  void set_in_channel(const int64_t in_channel);

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;

  /// \brief Method to get in_channel attribute.
  ///
  /// \return the number of input channel.
  int64_t get_in_channel() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CONV2D_BACKPROP_FILTER_FUSION_H_
