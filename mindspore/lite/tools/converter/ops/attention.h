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
#ifndef LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_ATTENTION_H_
#define LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_ATTENTION_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAttention = "Attention";
class Attention : public PrimitiveC {
 public:
  Attention() : PrimitiveC(kNameAttention) {
    InitIOName({"query", "key", "value", "w_q", "b_q", "w_k", "b_k", "w_v", "b_v", "w_o", "b_o"}, {"output"});
  }
  ~Attention() = default;
  MS_DECLARE_PARENT(Attention, PrimitiveC);
  void Init(const int64_t number_heads = 0, const int64_t key_dim = 0, const int64_t value_dim = 0);
  void set_num_heads(const int64_t num_heads);
  void set_key_dim(const int64_t key_dim);
  void set_value_dim(const int64_t value_dim);
  int64_t get_num_heads() const;
  int64_t get_key_dim() const;
  int64_t get_value_dim() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_ATTENTION_H_
