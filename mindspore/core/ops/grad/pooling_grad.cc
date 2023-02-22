/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(PoolingGrad, BaseOperator);
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
  (void)this->AddAttr(kPoolMode, api::MakeValue(swi));
}

PoolMode PoolingGrad::get_pool_mode() const {
  auto value_ptr = GetAttr(kPoolMode);
  return PoolMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_window(const std::vector<int64_t> &window) {
  (void)this->AddAttr(kWindow, api::MakeValue(window));
}

std::vector<int64_t> PoolingGrad::get_window() const {
  auto value_ptr = GetAttr(kWindow);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_stride(const std::vector<int64_t> &stride) {
  (void)this->AddAttr(kStride, api::MakeValue(stride));
}

std::vector<int64_t> PoolingGrad::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode PoolingGrad::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> PoolingGrad::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_round_mode(const RoundMode &round_mode) {
  int64_t swi = round_mode;
  (void)this->AddAttr(kRoundMode, api::MakeValue(swi));
}

RoundMode PoolingGrad::get_round_mode() const {
  auto value_ptr = GetAttr(kRoundMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return RoundMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

Format PoolingGrad::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_global(const bool global) { (void)this->AddAttr(kGlobal, api::MakeValue(global)); }

bool PoolingGrad::get_global() const {
  auto value_ptr = GetAttr(kGlobal);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNamePoolingGrad, PoolingGrad);
}  // namespace ops
}  // namespace mindspore
