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

#include <vector>

#include "ops/grad/de_conv2d_grad_filter.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void DeConv2DGradFilter::Init(const int64_t in_channel, const int64_t out_channel,
                              const std::vector<int64_t> &kernel_size, const PadMode &pad_mode,
                              const std::vector<int64_t> &pad_list, const std::vector<int64_t> &stride,
                              const std::vector<int64_t> &dilation, const int64_t group, const Format &format,
                              const ActivationType &activation_type, const bool has_bias) {
  set_in_channel(in_channel);
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
  set_activation_type(activation_type);
  set_has_bias(has_bias);
}

void DeConv2DGradFilter::set_in_channel(const int64_t in_channel) { this->AddAttr(kInChannel, MakeValue(in_channel)); }

int64_t DeConv2DGradFilter::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  return GetValue<int64_t>(value_ptr);
}

void DeConv2DGradFilter::set_out_channel(const int64_t out_channel) {
  this->AddAttr(kOutChannel, MakeValue(out_channel));
}

int64_t DeConv2DGradFilter::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

void DeConv2DGradFilter::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  this->AddAttr(kKernelSize, MakeValue(kernel_size));
}

std::vector<int64_t> DeConv2DGradFilter::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void DeConv2DGradFilter::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  this->AddAttr(kPadMode, MakeValue(swi));
}

PadMode DeConv2DGradFilter::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void DeConv2DGradFilter::set_pad_list(const std::vector<int64_t> &pad_list) {
  this->AddAttr(kPadList, MakeValue(pad_list));
}

std::vector<int64_t> DeConv2DGradFilter::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void DeConv2DGradFilter::set_stride(const std::vector<int64_t> &stride) { this->AddAttr(kStride, MakeValue(stride)); }

std::vector<int64_t> DeConv2DGradFilter::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void DeConv2DGradFilter::set_dilation(const std::vector<int64_t> &dilation) {
  this->AddAttr(kDilation, MakeValue(dilation));
}

std::vector<int64_t> DeConv2DGradFilter::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void DeConv2DGradFilter::set_group(const int64_t group) { this->AddAttr(kGroup, MakeValue(group)); }

int64_t DeConv2DGradFilter::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

void DeConv2DGradFilter::set_format(const Format &format) {
  int64_t swi = format;
  this->AddAttr(kFormat, MakeValue(swi));
}

Format DeConv2DGradFilter::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

void DeConv2DGradFilter::set_activation_type(const ActivationType &activation_type) {
  int64_t swi = activation_type;
  this->AddAttr(kActivationType, MakeValue(swi));
}

ActivationType DeConv2DGradFilter::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

void DeConv2DGradFilter::set_has_bias(const bool has_bias) { this->AddAttr(kHasBias, MakeValue(has_bias)); }

bool DeConv2DGradFilter::get_has_bias() const {
  auto value_ptr = GetAttr(kHasBias);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameDeConv2DGradFilter, DeConv2DGradFilter);
}  // namespace ops
}  // namespace mindspore
