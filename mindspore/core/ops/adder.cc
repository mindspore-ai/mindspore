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

#include "ops/adder.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Adder, BaseOperator);
void Adder::Init(const int64_t in_channel, const int64_t out_channel, const std::vector<int64_t> &kernel_size,
                 const PadMode &pad_mode, const std::vector<int64_t> &stride, const std::vector<int64_t> &pad_list,
                 const std::vector<int64_t> &dilation, const int64_t group, const Format &format) {
  set_in_channel(in_channel);
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_pad_mode(pad_mode);
  set_stride(stride);
  set_pad_list(pad_list);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
}

void Adder::set_in_channel(const int64_t in_channel) { (void)this->AddAttr(kInChannel, api::MakeValue(in_channel)); }

int64_t Adder::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  return GetValue<int64_t>(value_ptr);
}

void Adder::set_out_channel(const int64_t out_channel) {
  (void)this->AddAttr(kOutChannel, api::MakeValue(out_channel));
}

int64_t Adder::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

void Adder::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

std::vector<int64_t> Adder::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Adder::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode Adder::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void Adder::set_stride(const std::vector<int64_t> &stride) { (void)this->AddAttr(kStride, api::MakeValue(stride)); }

std::vector<int64_t> Adder::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Adder::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> Adder::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Adder::set_dilation(const std::vector<int64_t> &dilation) {
  (void)this->AddAttr(kDilation, api::MakeValue(dilation));
}

std::vector<int64_t> Adder::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Adder::set_group(const int64_t group) { (void)this->AddAttr(kGroup, api::MakeValue(group)); }

int64_t Adder::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

void Adder::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

Format Adder::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameAdder, Adder);
}  // namespace ops
}  // namespace mindspore
