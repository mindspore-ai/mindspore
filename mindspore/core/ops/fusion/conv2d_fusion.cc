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
#include <string>
#include <vector>

#include "ops/fusion/conv2d_fusion.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void Conv2DFusion::Init(int64_t in_channel, int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode,
                        const PadMode &pad_mode, const std::vector<int64_t> &pad, const std::vector<int64_t> &stride,
                        const std::vector<int64_t> &dilation, int64_t group, const Format &format,
                        const std::vector<int64_t> &pad_list, const ActivationType &activation_type) {
  this->set_in_channel(in_channel);
  this->set_out_channel(out_channel);
  this->set_kernel_size(kernel_size);
  this->set_mode(mode);
  this->set_pad_mode(pad_mode);
  this->set_pad(pad);
  this->set_stride(stride);
  this->set_dilation(dilation);
  this->set_group(group);
  this->set_format(format);
  this->set_pad_list(pad_list);
  this->set_activation_type(activation_type);
}
void Conv2DFusion::set_in_channel(const int64_t in_channel) { this->AddAttr(kInChannel, MakeValue(in_channel)); }
void Conv2DFusion::set_pad_list(const std::vector<int64_t> &pad_list) { this->AddAttr(kPadList, MakeValue(pad_list)); }
void Conv2DFusion::set_activation_type(const ActivationType &activation_type) {
  int64_t swi;
  swi = activation_type;
  this->AddAttr(kActivationType, MakeValue(swi));
}
int64_t Conv2DFusion::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  return GetValue<int64_t>(value_ptr);
}
std::vector<int64_t> Conv2DFusion::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
ActivationType Conv2DFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameConv2DFusion, Conv2DFusion);
}  // namespace ops
}  // namespace mindspore
