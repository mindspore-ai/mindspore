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

#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/grad/conv2d_backprop_input.h"

namespace mindspore {
namespace ops {
void Conv2DBackpropInput::Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode,
                               const PadMode &pad_mode, const std::vector<int64_t> &pad,
                               const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t group,
                               const Format &format, const std::vector<int64_t> &pad_list) {
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_mode(mode);
  set_pad_mode(pad_mode);
  set_pad(pad);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
  set_pad_list(pad_list);
}

void Conv2DBackpropInput::set_out_channel(int64_t out_channel) {
  AddAttr(kOutChannel,
          MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv2DBackpropInput::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  AddAttr(kKernelSize, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, name())));
}

void Conv2DBackpropInput::set_stride(const std::vector<int64_t> &stride) {
  AddAttr(kStride, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStride, stride, name(), true, true)));
}

void Conv2DBackpropInput::set_dilation(const std::vector<int64_t> &dilation) {
  AddAttr(kDilation, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, name(), true, true)));
}

void Conv2DBackpropInput::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, "zeros_list", 0, name());
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, "zeros_list", {0, 0, 0, 0}, name());
  }
  int64_t swi = pad_mode;
  AddAttr(kPadMode, MakeValue(swi));
}

void Conv2DBackpropInput::set_pad(const std::vector<int64_t> &pad) {
  CheckAndConvertUtils::CheckInteger("pad_size", pad.size(), kEqual, 4, name());
  AddAttr(kPad, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name(), true, true)));
}

void Conv2DBackpropInput::set_mode(int64_t mode) {
  AddAttr(kMode, MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv2DBackpropInput::set_group(int64_t group) {
  AddAttr(kGroup, MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

void Conv2DBackpropInput::set_format(const Format &format) {
  int64_t f = format;
  AddAttr(kFormat, MakeValue(f));
}

void Conv2DBackpropInput::set_pad_list(const std::vector<int64_t> &pad_list) {
  this->AddAttr(kPadList, MakeValue(pad_list));
}

int64_t Conv2DBackpropInput::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv2DBackpropInput::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2DBackpropInput::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2DBackpropInput::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv2DBackpropInput::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2DBackpropInput::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv2DBackpropInput::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv2DBackpropInput::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

Format Conv2DBackpropInput::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2DBackpropInput::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameConv2DBackpropInput, Conv2DBackpropInput);
}  // namespace ops
}  // namespace mindspore
