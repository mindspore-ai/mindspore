/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/prompt_flash_attention.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"

namespace mindspore {
namespace ops {

void PromptFlashAttention::Init(const int64_t num_heads, const int64_t pre_tokens, const int64_t next_tokens,
                                const std::string input_layout, const int64_t num_key_value_heads,
                                const float scale_value) {
  MS_LOG(INFO) << "Prompt Flash Attention init.";
  set_num_heads(num_heads);
  set_pre_tokens(pre_tokens);
  set_next_tokens(next_tokens);
  set_input_layout(input_layout);
  set_num_key_value_heads(num_key_value_heads);
  set_scale_value(scale_value);
}

void PromptFlashAttention::set_num_heads(const int64_t num_heads) {
  (void)this->AddAttr("num_heads", api::MakeValue(num_heads));
}

int64_t PromptFlashAttention::get_num_heads() const {
  auto value_ptr = GetAttr("num_heads");
  return GetValue<int64_t>(value_ptr);
}

void PromptFlashAttention::set_pre_tokens(const int64_t pre_tokens) {
  (void)this->AddAttr("pre_tokens", api::MakeValue(pre_tokens));
}

int64_t PromptFlashAttention::get_pre_tokens() const {
  auto value_ptr = GetAttr("pre_tokens");
  return GetValue<int64_t>(value_ptr);
}

void PromptFlashAttention::set_next_tokens(const int64_t next_tokens) {
  (void)this->AddAttr("next_tokens", api::MakeValue(next_tokens));
}

int64_t PromptFlashAttention::get_next_tokens() const {
  auto value_ptr = GetAttr("next_tokens");
  return GetValue<int64_t>(value_ptr);
}

void PromptFlashAttention::set_input_layout(const std::string input_layout) {
  (void)this->AddAttr("input_layout", api::MakeValue(input_layout));
}

std::string PromptFlashAttention::get_input_layout() const {
  auto value_ptr = GetAttr("input_layout");
  return GetValue<std::string>(value_ptr);
}

void PromptFlashAttention::set_num_key_value_heads(const int64_t num_key_value_heads) {
  (void)this->AddAttr("num_key_value_heads", api::MakeValue(num_key_value_heads));
}

int64_t PromptFlashAttention::get_num_key_value_heads() const {
  auto value_ptr = GetAttr("num_key_value_heads");
  return GetValue<int64_t>(value_ptr);
}

void PromptFlashAttention::set_scale_value(const float scale_value) {
  (void)this->AddAttr("scale_value", api::MakeValue(scale_value));
}

float PromptFlashAttention::get_scale_value() const {
  auto value_ptr = GetAttr("scale_value");
  return GetValue<float>(value_ptr);
}

MIND_API_OPERATOR_IMPL(PromptFlashAttention, BaseOperator);
REGISTER_PRIMITIVE_C(kNamePromptFlashAttention, PromptFlashAttention);
}  // namespace ops
}  // namespace mindspore
