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

#ifndef MINDSPORE_CORE_OPS_MATMUL_FFN_H_
#define MINDSPORE_CORE_OPS_MATMUL_FFN_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatmulFfn = "MatmulFfn";
/// \brief Computes the attentions of Feed Forward with hidden states
class MIND_API MatmulFfn : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MatmulFfn);
  /// \brief Constructor.
  MatmulFfn() : BaseOperator(kNameMatmulFfn) {
    InitIOName({"hidden_states", "weight_gate", "weight_up"}, {"output_gate", "output_up"});
  }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MATMUL_FFN_H_
