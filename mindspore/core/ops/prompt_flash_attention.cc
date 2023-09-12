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
                                const float scale_value) const {
  MS_LOG(INFO) << "Prompt Flash Attention init.";
  return;
}

MIND_API_OPERATOR_IMPL(PromptFlashAttention, BaseOperator);
REGISTER_PRIMITIVE_C(kNamePromptFlashAttention, PromptFlashAttention);
}  // namespace ops
}  // namespace mindspore
