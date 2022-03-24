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

#include "ops/max_pool.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MaxPool::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode MaxPool::get_pad_mode() const { return PadMode(GetValue<int64_t>(GetAttr(kPadMode))); }
void MaxPool::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(
    kKernelSize, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, this->name())));
}

std::vector<int64_t> MaxPool::get_kernel_size() const { return GetValue<std::vector<int64_t>>(GetAttr(kKernelSize)); }
void MaxPool::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr(kStrides,
                      api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, this->name())));
}

std::vector<int64_t> MaxPool::get_strides() const { return GetValue<std::vector<int64_t>>(GetAttr(kStrides)); }

void MaxPool::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}

Format MaxPool::get_format() const { return Format(GetValue<int64_t>(GetAttr(kFormat))); }

void MaxPool::set_pad(const std::vector<int64_t> &pad) { (void)this->AddAttr(kPad, api::MakeValue(pad)); }

std::vector<int64_t> MaxPool::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void MaxPool::set_round_mode(const RoundMode &round_mode) {
  int64_t swi = round_mode;
  (void)this->AddAttr(kRoundMode, api::MakeValue(swi));
}

RoundMode MaxPool::get_round_mode() const {
  auto value_ptr = GetAttr(kRoundMode);
  return RoundMode(GetValue<int64_t>(value_ptr));
}

void MaxPool::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride, const PadMode &pad_mode,
                   const Format &format, const std::vector<int64_t> &pad, const RoundMode &round_mode) {
  this->set_pad_mode(pad_mode);
  this->set_kernel_size(kernel_size);
  this->set_strides(stride);
  this->set_format(format);
  this->set_pad(pad);
  this->set_round_mode(round_mode);
}

MIND_API_BASE_IMPL(MaxPool, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_C(kNameMaxPool, MaxPool);
}  // namespace ops
}  // namespace mindspore
