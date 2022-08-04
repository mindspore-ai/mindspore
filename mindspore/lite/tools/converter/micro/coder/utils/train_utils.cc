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

#include "utils/train_utils.h"
#include <string>
#include <vector>
#include "src/common/prim_util.h"

namespace mindspore::lite::micro {
namespace {
constexpr char kGradName[] = "Gradients";
}
bool IsLossCoder(const OperatorCoder *coder) {
  MS_CHECK_TRUE_MSG(coder != nullptr, false, "coder is nullptr");
  bool is_loss = false;
  const std::vector<std::string> loss_names = {"loss_fct", "_loss_fn", "SigmoidCrossEntropy"};
  for (auto &name : loss_names) {
    if (coder->name().find(name) != std::string::npos) {
      is_loss = true;
      break;
    }
  }
  return is_loss;
}

bool IsGradCoder(const OperatorCoder *coder) {
  MS_CHECK_TRUE_MSG(coder != nullptr, false, "coder is nullptr");
  return coder->name().find(kGradName) != std::string::npos;
}

bool IsOptimizer(const OperatorCoder *coder) {
  MS_CHECK_TRUE_MSG(coder != nullptr, false, "coder is nullptr");
  auto node = coder->node();
  MS_CHECK_TRUE_MSG(node != nullptr, false, "coder's node is nullptr");
  auto node_type = static_cast<schema::PrimitiveType>(GetPrimitiveType(node->primitive_));
  return (node_type == schema::PrimitiveType_Adam) || (node_type == schema::PrimitiveType_SGD) ||
         (node_type == schema::PrimitiveType_ApplyMomentum);
}

bool IsMaskOutput(const OperatorCoder *coder) {
  MS_CHECK_TRUE_MSG(coder != nullptr, false, "coder is nullptr");
  auto node = coder->node();
  MS_CHECK_TRUE_MSG(node != nullptr, false, "coder's node is nullptr");
  auto node_type = static_cast<schema::PrimitiveType>(GetPrimitiveType(node->primitive_));
  return (IsOptimizer(coder) || (node_type == schema::PrimitiveType_Assign));
}
}  // namespace mindspore::lite::micro
