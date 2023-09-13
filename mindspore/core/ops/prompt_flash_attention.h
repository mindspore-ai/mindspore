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

#ifndef MINDSPORE_CORE_OPS_PROMPT_FLASH_ATTENTION_H
#define MINDSPORE_CORE_OPS_PROMPT_FLASH_ATTENTION_H
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePromptFlashAttention = "PromptFlashAttention";
/// \brief Computes a tensor to the power of the second input.
/// Refer to Python API @ref mindspore.ops.PromptFlashAttention for more details.
class MIND_API PromptFlashAttention : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PromptFlashAttention);
  /// \brief Constructor.
  PromptFlashAttention() : BaseOperator(kNamePromptFlashAttention) {
    InitIOName({"query", "key", "value", "atten_mask"}, {"attention_out"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.PromptFlashAttention for the inputs.
  void Init(const int64_t num_heads, const int64_t pre_tokens, const int64_t next_tokens,
            const std::string input_layout, const int64_t num_key_value_heads, const float scale_value);

  /// \brief Set num_heads.
  void set_num_heads(const int64_t num_heads);

  /// \brief Get num_heads.
  ///
  /// \return num_heads.
  int64_t get_num_heads() const;

  /// \brief Set pre_tokens.
  void set_pre_tokens(const int64_t pre_tokens);

  /// \brief Get pre_tokens.
  ///
  /// \return pre_tokens.
  int64_t get_pre_tokens() const;

  /// \brief Set next_tokens.
  void set_next_tokens(const int64_t next_tokens);

  /// \brief Get next_tokens.
  ///
  /// \return next_tokens.
  int64_t get_next_tokens() const;

  /// \brief Set input_layout.
  void set_input_layout(const std::string input_layout);

  /// \brief Get input_layout.
  ///
  /// \return input_layout.
  std::string get_input_layout() const;

  /// \brief Set num_key_value_heads.
  void set_num_key_value_heads(const int64_t num_key_value_heads);

  /// \brief Get num_key_value_heads.
  ///
  /// \return num_key_value_heads.
  int64_t get_num_key_value_heads() const;

  /// \brief Set scale_value.
  void set_scale_value(const float scale_value);

  /// \brief Get scale_value.
  ///
  /// \return scale_value.
  float get_scale_value() const;
};
MIND_API abstract::AbstractBasePtr PromptFlashAttentionInfer(const abstract::AnalysisEnginePtr &,
                                                             const PrimitivePtr &primitive,
                                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimPromptFlashAttentionPtr = std::shared_ptr<PromptFlashAttention>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PROMPT_FLAPromptFlashAttentionSH_ATTENTION_H
