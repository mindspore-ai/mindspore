
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
#include <memory>

#include "tools/converter/ops/attention.h"
#include "ops/op_utils.h"

namespace mindspore::ops {
void Attention::Init(int64_t number_heads, int64_t key_dim, int64_t value_dim) {
  this->set_num_heads(number_heads);
  this->set_key_dim(key_dim);
  this->set_value_dim(value_dim);
}
void Attention::set_num_heads(const int64_t num_heads) { this->AddAttr(kNumHeads, MakeValue(num_heads)); }
void Attention::set_key_dim(const int64_t key_dim) { this->AddAttr(kKeyDim, MakeValue(key_dim)); }
void Attention::set_value_dim(const int64_t value_dim) { this->AddAttr(kValueDim, MakeValue(value_dim)); }
int64_t Attention::get_num_heads() const {
  auto value_ptr = this->GetAttr(kNumHeads);
  return GetValue<int64_t>(value_ptr);
}
int64_t Attention::get_key_dim() const {
  auto value_ptr = this->GetAttr(kKeyDim);
  return GetValue<int64_t>(value_ptr);
}
int64_t Attention::get_value_dim() const {
  auto value_ptr = this->GetAttr(kValueDim);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameAttention, Attention);
}  // namespace mindspore::ops
