/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_CFG_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_CFG_H_

#include <vector>
#include <string>

namespace mindspore {
namespace lite {
class ConvConfig {
 public:
  ConvConfig() = default;
  int in_channel_ = 3;                          /**< The channel number of the input of the Conv2d layer */
  int out_channel_ = 3;                         /**< The channel number of the output tensor of the Conv2d layer */
  std::vector<int64_t> kernel_size_ = {3, 3};   /**< Specifies the height and width of the 2D convolution kernel. */
  std::vector<int64_t> stride_ = {1, 1};        /**< The movement stride of the 2D convolution kernel */
  std::vector<int64_t> padding_ = {0, 0, 0, 0}; /**< The top, bottom, left, and right padding input */
  std::vector<int64_t> dilation_ = {1, 1};      /**< diletion height and width*/
  int group_ = 1;                               // < Splits filter into groups, `in_channels` and `out_channels` must be
                   // divisible by `group`. If the group is equal to `in_channels` and `out_channels`,
                   // this 2D convolution layer also can be called 2D depthwise convolution layer */
  bool has_bias = false; /** < Whether the Conv2d layer has a bias parameter */
  std::string weight_init_ =
    "normal"; /**< Initialization method of weight parameter ("normal","uniform", "ones", "zeros") */
  std::string pad_mode_ = "same"; /**<  Specifies padding mode. The optional values are "same", "valid", "pad" */

 private:
  std::string bias_init_ = "zeros";
  std::string data_format;
};

class DenseConfig {
 public:
  int in_channels_;       /**< The number of channels in the input space */
  int out_channels_;      /**< The number of channels in the output space */
  bool has_bias_ = false; /** Specifies whether the layer uses a bias vector **/
 private:
  std::string weight_init_ = "normal";
  std::string bias_init_ = "zeros";
  std::string activation_ = "none";
};

class PoolingConfig {
 public:
  PoolingConfig() = default;
  std::vector<int64_t> kernel_size_ = {1, 1}; /**< Specifies the height and width of the 2D kernel. */
  std::vector<int64_t> stride_ = {1, 1};      /**< The movement stride of the 2D kernel */
  std::string pad_mode_ = "same";             /**<  Specifies padding mode. The optional values are "same", "valid" */
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXPRESSION_CFG_H_
