
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

#include "ops/encoder_layer.h"

#include "ops/primitive_c.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore::ops {
MIND_API_OPERATOR_IMPL(EncoderLayer, BaseOperator);

void EncoderLayer::set_head_num(int64_t head_num) {
  (void)this->AddAttr(kEncoderLayerNumHeads, api::MakeValue(head_num));
}

void EncoderLayer::set_head_size(int64_t head_size) {
  (void)this->AddAttr(kEncoderLayerSizePerHead, api::MakeValue(head_size));
}

void EncoderLayer::set_post_layernorm(bool post_layernorm) {
  (void)this->AddAttr(kEncoderLayerPostLayernorm, api::MakeValue(post_layernorm));
}
void EncoderLayer::set_eps_layernorm1(float eps_layernorm1) {
  (void)this->AddAttr(kEncoderLayerEpsLayerNorm1, api::MakeValue(eps_layernorm1));
}
void EncoderLayer::set_eps_layernorm2(float eps_layernorm2) {
  (void)this->AddAttr(kEncoderLayerEpsLayerNorm2, api::MakeValue(eps_layernorm2));
}
void EncoderLayer::set_ffn_hidden_size(int64_t ffn_hidden_size) {
  (void)this->AddAttr(kEncoderLayerFfnHiddenSize, api::MakeValue(ffn_hidden_size));
}
void EncoderLayer::set_position_bias(bool position_bias) {
  (void)this->AddAttr(kPositionBias, api::MakeValue(position_bias));
}

int64_t EncoderLayer::get_head_num() const {
  auto value_ptr = this->GetAttr(kEncoderLayerNumHeads);
  return GetValue<int64_t>(value_ptr);
}

int64_t EncoderLayer::get_head_size() const {
  auto value_ptr = this->GetAttr(kEncoderLayerSizePerHead);
  return GetValue<int64_t>(value_ptr);
}

bool EncoderLayer::get_post_layernorm() const {
  auto value_ptr = this->GetAttr(kEncoderLayerPostLayernorm);
  return GetValue<bool>(value_ptr);
}
float EncoderLayer::get_eps_layernorm1() const {
  auto value_ptr = this->GetAttr(kEncoderLayerEpsLayerNorm1);
  return GetValue<float>(value_ptr);
}
float EncoderLayer::get_eps_layernorm2() const {
  auto value_ptr = this->GetAttr(kEncoderLayerEpsLayerNorm2);
  return GetValue<float>(value_ptr);
}
int64_t EncoderLayer::get_ffn_hidden_size() const {
  auto value_ptr = this->GetAttr(kEncoderLayerFfnHiddenSize);
  return GetValue<int64_t>(value_ptr);
}
bool EncoderLayer::get_position_bias() const {
  auto value_ptr = this->GetAttr(kPositionBias);
  return GetValue<bool>(value_ptr);
}

void EncoderLayer::Init(int64_t head_num, int64_t head_size, float eps_layernorm1, float eps_layernorm2,
                        int64_t ffn_hidden_size, bool position_bias, bool post_layernorm = false) {
  this->set_head_num(head_num);
  this->set_head_size(head_size);
  this->set_post_layernorm(post_layernorm);
  this->set_eps_layernorm1(eps_layernorm1);
  this->set_eps_layernorm2(eps_layernorm2);
  this->set_ffn_hidden_size(ffn_hidden_size);
  this->set_position_bias(position_bias);
}
REGISTER_PRIMITIVE_C(kNameEncoderLayer, EncoderLayer);
}  // namespace mindspore::ops
