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

#include "ops/fusion/depthwise_conv2d_fusion.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void DepthWiseConv2DFusion::Init(const int64_t channel_multiplier, const std::vector<int64_t> &kernel_size,
                                 const int64_t mode, const PadMode &pad_mode, const std::vector<int64_t> &pad,
                                 const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                 const int64_t group, const ActivationType &activation_type) {
  auto prim_name = this->name();
  this->set_format(NCHW);
  this->AddAttr("offset_a", MakeValue(0));
  this->set_mode(CheckAndConvertUtils::CheckInteger("mode", mode, kEqual, 3, prim_name));

  this->set_kernel_size(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, prim_name));
  auto strides = CheckAndConvertUtils::CheckPositiveVector(kStride, stride, this->name(), false, false);
  if (strides[0] != strides[1]) {
    MS_EXCEPTION(ValueError) << "The height and width of stride should be equal, but got height " << strides[0]
                             << ", width " << strides[1];
  }
  this->set_stride(strides);
  auto dilations = CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, this->name(), false, false);
  if (dilations[0] != dilations[1]) {
    MS_EXCEPTION(ValueError) << "The height and width of dilation should be equal, but got height " << dilations[0]
                             << ", width " << dilations[1];
  }
  this->set_dilation(dilations);
  this->set_pad_mode(pad_mode);

  CheckAndConvertUtils::CheckInteger("pad_size", pad.size(), kEqual, 4, prim_name);
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check("pad_item", item, kGreaterEqual, "zeros_list", 0, prim_name);
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, "zeros_list", {0, 0, 0, 0}, prim_name);
  }
  this->set_pad(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, this->name(), true, true));

  this->set_out_channel(
    CheckAndConvertUtils::CheckInteger("channel_multiplier", channel_multiplier, kGreaterThan, 0, prim_name));
  this->set_group(CheckAndConvertUtils::CheckInteger("group", group, kGreaterThan, 0, prim_name));
  this->set_activation_type(activation_type);
}

void DepthWiseConv2DFusion::set_activation_type(const ActivationType &activation_type) {
  int64_t swi;
  swi = activation_type;
  this->AddAttr(kActivationType, MakeValue(swi));
}

ActivationType DepthWiseConv2DFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameDepthWiseConv2DFusion, DepthWiseConv2DFusion);
}  // namespace ops
}  // namespace mindspore
