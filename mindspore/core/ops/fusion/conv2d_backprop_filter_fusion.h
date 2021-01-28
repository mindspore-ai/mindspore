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

#ifndef MINDSPORE_CORE_OPS_CONV2D_BACKPROP_FILTER_FUSION_H_
#define MINDSPORE_CORE_OPS_CONV2D_BACKPROP_FILTER_FUSION_H_
#include <vector>
#include <memory>

#include "ops/grad/conv2d_backprop_filter.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConv2DBackpropFilterFusion = "Conv2DBackpropFilterFusion";
class Conv2DBackpropFilterFusion : public Conv2DBackpropFilter {
 public:
  Conv2DBackpropFilterFusion() : Conv2DBackpropFilter(kNameConv2DBackpropFilterFusion) {
    InitIOName({"out_backprop", "input", "filter_sizes"}, {"output"});
  }
  ~Conv2DBackpropFilterFusion() = default;
  MS_DECLARE_PARENT(Conv2DBackpropFilterFusion, Conv2DBackpropFilter);
  void Init(const int64_t in_channel, const int64_t out_channel, const std::vector<int64_t> &kernel_size,
            const PadMode &pad_mode = VALID, const std::vector<int64_t> &pad_list = {0, 0, 0, 0},
            const int64_t mode = 1, const std::vector<int64_t> &stride = {1, 1},
            const std::vector<int64_t> &dilation = {1, 1, 1, 1}, const int64_t group = 1, const Format &format = NCHW,
            const ActivationType activation_type = NO_ACTIVATION);
  void set_activation_type(const ActivationType activation_type);
  void set_in_channel(const int64_t in_channel);

  ActivationType get_activation_type() const;
  int64_t get_in_channel() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CONV2D_BACKPROP_FILTER_FUSION_H_
