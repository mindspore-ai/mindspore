/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_GRAD_RELU_GRAD_V2_H_
#define MINDSPORE_CORE_OPS_GRAD_RELU_GRAD_V2_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kReluGradV2 = "ReluGradV2";
/// \brief Grad op of ReLUV2.
class MIND_API ReluGradV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReluGradV2);
  /// \brief Constructor.
  ReluGradV2() : BaseOperator(kReluGradV2) { InitIOName({"gradients", "mask"}, {"output"}); }
  /// \brief Constructor.
  explicit ReluGradV2(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"gradients", "mask"}, {"output"});
  }
  /// \brief Init.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRAD_RELU_GRAD_V2_H_
