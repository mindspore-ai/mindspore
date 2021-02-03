/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_DEPTHWISE_CONV2D_FUSION_H_
#define MINDSPORE_CORE_OPS_DEPTHWISE_CONV2D_FUSION_H_
#include <vector>

#include "ops/depthwise_conv2d.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDepthWiseConv2DFusion = "DepthWiseConv2DFusion";
class DepthWiseConv2DFusion : public DepthWiseConv2D {
 public:
  MS_DECLARE_PARENT(DepthWiseConv2DFusion, DepthWiseConv2D);
  void Init(const int64_t out_channel, const std::vector<int64_t> &kernel_size, const int64_t mode = 1,
            const PadMode &pad_mode = VALID, const std::vector<int64_t> &pad = {0, 0, 0, 0},
            const std::vector<int64_t> &stride = {1, 1, 1, 1}, const std::vector<int64_t> &dilation = {1, 1, 1, 1},
            const int64_t group = 1, const ActivationType &activation_type = NO_ACTIVATION);
  void set_activation_type(const ActivationType &activation_type);
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DEPTHWISE_CONV2D_FUSION_H_
