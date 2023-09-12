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

#ifndef MINDSPORE_CORE_OPS_INCRE_FlashAttentionSH_ATTENTION_H
#define MINDSPORE_CORE_OPS_INCRE_FlashAttentionSH_ATTENTION_H
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameIncreFlashAttention = "IncreFlashAttention";
/// \brief Computes a tensor to the power of the second input.
/// Refer to Python API @ref mindspore.ops.IncreFlashAttention for more details.
class MIND_API IncreFlashAttention : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(IncreFlashAttention);
  /// \brief Constructor.
  IncreFlashAttention() : BaseOperator(kNameIncreFlashAttention) {
    InitIOName({"query", "key", "value", "atten_mask"}, {"attention_out"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.IncreFlashAttention for the inputs.

  void Init() const;
};
MIND_API abstract::AbstractBasePtr IncreFlashAttentionInfer(const abstract::AnalysisEnginePtr &,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimIncreFlashAttentionPtr = std::shared_ptr<IncreFlashAttention>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INCRE_FlashAttentionSH_ATTENTION_H
