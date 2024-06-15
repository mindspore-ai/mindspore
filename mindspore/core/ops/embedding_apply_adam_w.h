/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_EMBEDDING_APPLY_ADAM_W_H
#define MINDSPORE_CORE_OPS_EMBEDDING_APPLY_ADAM_W_H

#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameEmbeddingApplyAdamW = "EmbeddingApplyAdamW";
class MIND_API EmbeddingApplyAdamW : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EmbeddingApplyAdamW);
  EmbeddingApplyAdamW() : BaseOperator(kNameEmbeddingApplyAdamW) {
    InitIOName({"var_handle", "beta1_power", "beta2_power", "lr", "weight_decay", "beta1", "beta2", "epsilon", "grad",
                "keys", "max_grad_norm", "global_step"},
               {"var_handle"});
  }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EMBEDDING_APPLY_ADAM_W_H
