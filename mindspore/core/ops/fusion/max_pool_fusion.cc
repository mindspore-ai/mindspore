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

#include "ops/fusion/max_pool_fusion.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MaxPoolFusion::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride,
                         const PadMode &pad_mode, const Format &format, const std::vector<int64_t> &pad,
                         const RoundMode &round_mode, const bool global, const ActivationType activation_type) {
  this->set_pad_mode(pad_mode);
  this->set_kernel_size(kernel_size);
  this->set_strides(stride);
  this->set_format(format);
  this->set_pad(pad);
  this->set_round_mode(round_mode);
  this->set_global(global);
  this->set_activation_type(activation_type);
}

void MaxPoolFusion::set_global(const bool global) { (void)AddAttr(kGlobal, api::MakeValue(global)); }

void MaxPoolFusion::set_activation_type(ActivationType activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}

bool MaxPoolFusion::get_global() const {
  auto value_ptr = GetAttr(kGlobal);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

ActivationType MaxPoolFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

MIND_API_OPERATOR_IMPL(MaxPoolFusion, MaxPool);
REGISTER_PRIMITIVE_C(kNameMaxPoolFusion, MaxPoolFusion);
}  // namespace ops
}  // namespace mindspore
