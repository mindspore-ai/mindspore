/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "c_ops/conv2d.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace {
constexpr auto kKernelSize = "kernel_size";
constexpr auto kStride = "stride";
constexpr auto kDilation = "dilation";
constexpr auto kPadMode = "pad_mode";
constexpr auto kPad = "pad";
constexpr auto kMode = "mode";
constexpr auto kGroup = "group";
constexpr auto kOutputChannel = "output channel";
constexpr auto kPadList = "pad_list";
constexpr auto kConv2DName = "Conv2D";
}  // namespace
Conv2d::Conv2d() : PrimitiveC(kConv2DName) { InitIOName({"x", "w"}, {"output"}); }

void Conv2d::Init(int out_channel, const std::vector<int> &kernel_size, int mode, const std::string &pad_mode,
                  const std::vector<int> &pad, const std::vector<int> &stride, const std::vector<int> &dilation,
                  int group) {
  auto prim_name = this->name();
  this->AddAttr("data_format", MakeValue("NCHW"));
  this->AddAttr("offset_a", MakeValue(0));
  this->SetKernelSize(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, prim_name));
  this->SetStride(CheckAndConvertUtils::CheckPositiveVector(kStride, stride, this->name(), true, true));
  this->SetDilation(CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, this->name(), true, true));
  this->SetPadMode(CheckAndConvertUtils::CheckString(kPadMode, pad_mode, {"valid", "same", "pad"}, prim_name));
  CheckAndConvertUtils::CheckInteger("pad size", pad.size(), kEqual, 4, prim_name);
  if (pad_mode == "pad") {
    for (auto item : pad) {
      CheckAndConvertUtils::Check("pad item", item, kGreaterEqual, "zeros list", 0, prim_name);
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, "zeros list", {0, 0, 0, 0}, prim_name);
  }
  this->SetPad(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, this->name(), true, true));
  this->SetMode(CheckAndConvertUtils::CheckInteger("mode", mode, kEqual, 1, prim_name));
  this->SetOutChannel(CheckAndConvertUtils::CheckInteger("out_channel", out_channel, kGreaterThan, 0, prim_name));
  this->SetGroup(CheckAndConvertUtils::CheckInteger("group", group, kGreaterThan, 0, prim_name));
}
std::vector<int> Conv2d::GetKernelSize() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int>>(value_ptr);
}
std::vector<int> Conv2d::GetStride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int>>(value_ptr);
}
std::vector<int> Conv2d::GetDilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int>>(value_ptr);
}
std::string Conv2d::GetPadMode() const {
  auto value_ptr = this->GetAttr(kPadMode);
  return GetValue<string>(value_ptr);
}
std::vector<int> Conv2d::GetPad() const {
  auto value_ptr = this->GetAttr(kPad);
  return GetValue<std::vector<int>>(value_ptr);
}
int Conv2d::GetMode() const {
  auto value_ptr = this->GetAttr(kMode);
  return GetValue<int>(value_ptr);
}

int Conv2d::GetGroup() const {
  auto value_ptr = this->GetAttr(kGroup);
  return GetValue<int>(value_ptr);
}
int Conv2d::GetOutputChannel() const {
  auto value_ptr = this->GetAttr(kOutputChannel);
  return GetValue<int>(value_ptr);
}

void Conv2d::SetKernelSize(const std::vector<int> &kernel_size) { this->AddAttr(kKernelSize, MakeValue(kernel_size)); }
void Conv2d::SetStride(const std::vector<int> &stride) { this->AddAttr(kStride, MakeValue(stride)); }
void Conv2d::SetDilation(const std::vector<int> &dilation) { this->AddAttr(kDilation, MakeValue(dilation)); }
void Conv2d::SetPadMode(const std::string &pad_mode) { this->AddAttr(kPadMode, MakeValue(pad_mode)); }
void Conv2d::SetPad(const std::vector<int> &pad) { this->AddAttr(kPad, MakeValue(pad)); }
void Conv2d::SetMode(int mode) { this->AddAttr(kMode, MakeValue(mode)); }
void Conv2d::SetGroup(int group) { this->AddAttr(kGroup, MakeValue(group)); }
void Conv2d::SetOutChannel(int output_channel) { this->AddAttr(kOutputChannel, MakeValue(output_channel)); }
void Conv2d::SetPadList(const std::vector<int> &pad_list) { this->AddAttr(kPadList, MakeValue(pad_list)); }
}  // namespace mindspore
