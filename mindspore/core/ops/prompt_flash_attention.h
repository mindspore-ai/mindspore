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

#ifndef MINDSPORE_CORE_OPS_PROMPT_FLASH_ATTENTION_H_
#define MINDSPORE_CORE_OPS_PROMPT_FLASH_ATTENTION_H_
#include <map>
#include <memory>
#include <vector>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
namespace mindspore {
namespace ops {
constexpr auto kNamePromptFlashAttention = "PromptFlashAttention";
enum PromptFlashAttentionInputIndex : size_t {
  kPromptFlashAttentionInputQueryIndex = 0,
  kPromptFlashAttentionInputKeyIndex,
  kPromptFlashAttentionInputValueIndex,
  kPromptFlashAttentionInputAttnMaskIndex,
  kPromptFlashAttentionInputPaddingMaskIndex,
  kPromptFlashAttentionInputActualSeqLengthsIndex,
  kPromptFlashAttentionInputsNum,
};
enum PromptFlashAttentionOutputIndex : size_t {
  kPromptFlashAttentionOutputAttentionOutIndex = 0,
  kPromptFlashAttentionOutputsNum,
};

/// \brief PromptFlashAttention.
/// Refer to Python API @ref mindspore.ops.PromptFlashAttention for more details.
class MIND_API PromptFlashAttention : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PromptFlashAttention);
  /// \brief Constructor.
  PromptFlashAttention() : BaseOperator(kNamePromptFlashAttention) {
    InitIOName({"query", "key", "value", "attn_mask", "padding_mask", "actual_seq_lengths"}, {"attention_out"});
  }
};
AbstractBasePtr PromptFlashAttentionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args);
using PromptFlashAttentionPtr = std::shared_ptr<PromptFlashAttention>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_PROMPT_FLASH_ATTENTION_H_
