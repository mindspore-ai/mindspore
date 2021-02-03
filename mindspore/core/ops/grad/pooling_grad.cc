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

#include "ops/grad/pooling_grad.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void PoolingGrad::Init(const PoolMode &pool_mode, const std::vector<int64_t> &window,
                       const std::vector<int64_t> &stride, const PadMode &pad_mode,
                       const std::vector<int64_t> &pad_list, const RoundMode &round_mode, const Format &format,
                       const bool global) {
  set_pool_mode(pool_mode);
  set_window(window);
  set_stride(stride);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_round_mode(round_mode);
  set_format(format);
  set_global(global);
}

void PoolingGrad::set_pool_mode(const PoolMode &pool_mode) {
  int64_t swi = pool_mode;
  this->AddAttr(kPoolMode, MakeValue(swi));
}

PoolMode PoolingGrad::get_pool_mode() const {
  auto value_ptr = GetAttr(kPoolMode);
  return PoolMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_window(const std::vector<int64_t> &window) { this->AddAttr(kWindow, MakeValue(window)); }

std::vector<int64_t> PoolingGrad::get_window() const {
  auto value_ptr = GetAttr(kWindow);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_stride(const std::vector<int64_t> &stride) { this->AddAttr(kStride, MakeValue(stride)); }

std::vector<int64_t> PoolingGrad::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  this->AddAttr(kPadMode, MakeValue(swi));
}

PadMode PoolingGrad::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_pad_list(const std::vector<int64_t> &pad_list) { this->AddAttr(kPadList, MakeValue(pad_list)); }

std::vector<int64_t> PoolingGrad::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_round_mode(const RoundMode &round_mode) {
  int64_t swi = round_mode;
  this->AddAttr(kRoundMode, MakeValue(swi));
}

RoundMode PoolingGrad::get_round_mode() const {
  auto value_ptr = GetAttr(kRoundMode);
  return RoundMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_format(const Format &format) {
  int64_t swi = format;
  this->AddAttr(kFormat, MakeValue(swi));
}

Format PoolingGrad::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_global(const bool global) { this->AddAttr(kGlobal, MakeValue(global)); }

bool PoolingGrad::get_global() const {
  auto value_ptr = GetAttr(kGlobal);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNamePoolingGrad, PoolingGrad);
}  // namespace ops
}  // namespace mindspore
