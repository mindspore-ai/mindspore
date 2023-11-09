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

#include "ops/fusion/avg_pool_fusion.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
void AvgPoolFusion::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride,
                         const PadMode &pad_mode, const Format &format, const std::vector<int64_t> &pad,
                         const RoundMode &round_mode, const bool global, const ActivationType activation_type) {
  this->set_global(global);
  this->set_activation_type(activation_type);
}

void AvgPoolFusion::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr("kernel_size", api::MakeValue(kernel_size));
}

std::vector<int64_t> AvgPoolFusion::get_kernel_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr("kernel_size"));
}

void AvgPoolFusion::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr("strides", api::MakeValue(strides));
}

std::vector<int64_t> AvgPoolFusion::get_strides() const { return GetValue<std::vector<int64_t>>(GetAttr("strides")); }

void AvgPoolFusion::set_pad_mode(const int64_t &pad_mode) { (void)this->AddAttr("pad_mode", api::MakeValue(pad_mode)); }

int64_t AvgPoolFusion::get_pad_mode() const { return GetValue<int64_t>(GetAttr("pad_mode")); }

void AvgPoolFusion::set_data_format(const int64_t &data_format) {
  (void)this->AddAttr("data_format", api::MakeValue(data_format));
}

int64_t AvgPoolFusion::get_data_format() const { return GetValue<int64_t>(GetAttr("data_format")); }

void AvgPoolFusion::set_pad(const std::vector<int64_t> &pad) { (void)this->AddAttr("pad", api::MakeValue(pad)); }

std::vector<int64_t> AvgPoolFusion::get_pad() const { return GetValue<std::vector<int64_t>>(GetAttr("pad")); }

void AvgPoolFusion::set_round_mode(const int64_t &round_mode) {
  (void)this->AddAttr("round_mode", api::MakeValue(round_mode));
}

int64_t AvgPoolFusion::get_round_mode() const { return GetValue<int64_t>(GetAttr("round_mode")); }

void AvgPoolFusion::set_global(const bool global) { (void)AddAttr(kGlobal, api::MakeValue(global)); }

void AvgPoolFusion::set_activation_type(ActivationType activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}

bool AvgPoolFusion::get_global() const {
  auto value_ptr = GetAttr(kGlobal);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

ActivationType AvgPoolFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

MIND_API_OPERATOR_IMPL(AvgPoolFusion, BaseOperator);
REGISTER_PRIMITIVE_C(kNameAvgPoolFusion, AvgPoolFusion);
}  // namespace ops
}  // namespace mindspore
