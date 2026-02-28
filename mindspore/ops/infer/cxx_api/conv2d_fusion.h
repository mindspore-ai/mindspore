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

#ifndef MINDSPORE_CORE_OPS_CONV2D_FUSION_H_
#define MINDSPORE_CORE_OPS_CONV2D_FUSION_H_
#include <vector>

#include "mindapi/base/types.h"
#include "infer/conv2d.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConv2DFusion = "Conv2DFusion";
/// \brief Conv2DFusion defined Conv2D operator prototype of lite.
class OPS_API Conv2DFusion : public Conv2D {
 public:
  MIND_API_BASE_MEMBER(Conv2DFusion);
  /// \brief Constructor.
  Conv2DFusion() : Conv2D(kNameConv2DFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] in_channel Define the number of input channel.
  /// \param[in] out_channel Define the number of output channel.
  /// \param[in] kernel_size Define the size of the filter kernel.
  /// \param[in] mode Define the category of conv, which is useless on lite.
  /// \param[in] pad_mode Define the padding method.
  /// \param[in] pad Define the concrete padding value on H and W dimension, which is replaced with pad_list.
  /// \param[in] stride Define the moving size of the filter kernel.
  /// \param[in] dilation Define the coefficient of expansion of the filter kernel, which is useful for dilated
  ///            convolution.
  /// \param[in] group Define the number of group.
  /// \param[in] format Define the format of input tensor.
  /// \param[in] pad_list Define the concrete padding value on H and W dimension.
  /// \param[in] activation_type Define the activation type.
  void Init(int64_t in_channel, int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode = 1,
            const PadMode &pad_mode = VALID, const std::vector<int64_t> &pad = {0, 0, 0, 0},
            const std::vector<int64_t> &stride = {1, 1, 1, 1}, const std::vector<int64_t> &dilation = {1, 1, 1, 1},
            int64_t group = 1, const Format &format = NCHW, const std::vector<int64_t> &pad_list = {0, 0, 0, 0},
            const ActivationType &activation_type = NO_ACTIVATION);

  /// \brief Method to set in_channel attribute.
  ///
  /// \param[in] in_channel Define the number of input channel.
  void set_in_channel(const int64_t in_channel);

  /// \brief Method to set pad_list attribute.
  ///
  /// \param[in] pad_list Define the concrete padding value on H and W dimension.
  void set_pad_list(const std::vector<int64_t> &pad_list);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType &activation_type);

  /// \brief Method to get in_channel attribute.
  ///
  /// \return the number of input channel.
  int64_t get_in_channel() const;

  /// \brief Method to get pad_list attribute.
  ///
  /// \return padding value.
  std::vector<int64_t> get_pad_list() const;

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CONV2D_FUSION_H_
