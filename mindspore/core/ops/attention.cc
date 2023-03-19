
/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/attention.h"

#include "ops/primitive_c.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore::ops {
MIND_API_OPERATOR_IMPL(Attention, BaseOperator);

void Attention::set_head_num(int64_t head_num) { (void)this->AddAttr(kNumHeads, api::MakeValue(head_num)); }

void Attention::set_head_size(int64_t head_size) { (void)this->AddAttr(kSizePerHead, api::MakeValue(head_size)); }

void Attention::set_cross(bool cross) { (void)this->AddAttr(kCross, api::MakeValue(cross)); }

void Attention::set_position_bias(bool position_bias) {
  (void)this->AddAttr(kPositionBias1, api::MakeValue(position_bias));
}
void Attention::set_scale(float scale) { (void)this->AddAttr(kScale, api::MakeValue(scale)); }
int64_t Attention::get_head_num() const {
  auto value_ptr = this->GetAttr(kNumHeads);
  return GetValue<int64_t>(value_ptr);
}

int64_t Attention::get_head_size() const {
  auto value_ptr = this->GetAttr(kSizePerHead);
  return GetValue<int64_t>(value_ptr);
}

bool Attention::get_cross() const {
  auto value_ptr = this->GetAttr(kCross);
  return GetValue<bool>(value_ptr);
}

bool Attention::get_position_bias() const {
  auto value_ptr = this->GetAttr(kPositionBias1);
  return GetValue<bool>(value_ptr);
}
float Attention::get_scale() const {
  auto value_ptr = this->GetAttr(kScale);
  return GetValue<float>(value_ptr);
}
void Attention::Init(int64_t head_num, int64_t head_size, bool position_bias, bool cross, float scale) {
  this->set_head_num(head_num);
  this->set_head_size(head_size);
  this->set_cross(cross);
  this->set_position_bias(position_bias);
  this->set_scale(scale);
}
REGISTER_PRIMITIVE_C(kNameAttention, Attention);
}  // namespace mindspore::ops
