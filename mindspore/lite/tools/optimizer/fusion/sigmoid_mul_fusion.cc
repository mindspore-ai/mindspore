/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/sigmoid_mul_fusion.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "ops/op_utils.h"
#include "src/param_value_lite.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
namespace {
bool IsMulNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimMulFusion);
  }
  return false;
}
}  // namespace
const BaseRef SigmoidMulFusion::DefinePattern() const {
  auto input_var = std::make_shared<Var>();
  auto activation_var = std::make_shared<CondVar>(IsActivationNode);
  auto mul_var = std::make_shared<CondVar>(IsMulNode);
  auto activation_input = VectorRef({activation_var, input_var});
  return VectorRef({mul_var, input_var, activation_input});
}

// x * sigmoid(x) ->swish(x)
const AnfNodePtr SigmoidMulFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto mul_cnode = node->cast<CNodePtr>();
  MS_ASSERT(mul_cnode != nullptr);
  auto activation_cnode = mul_cnode->input(2)->cast<CNodePtr>();
  MS_ASSERT(activation_cnode != nullptr);
  // activation must sigmoid
  auto activation_prim = GetValueNode<std::shared_ptr<mindspore::ops::Activation>>(activation_cnode->input(0));
  if (activation_prim == nullptr || (activation_prim->GetAttr(ops::kActivationType) != nullptr &&
                                     activation_prim->get_activation_type() != mindspore::SIGMOID)) {
    return nullptr;
  }
  activation_prim->set_activation_type(mindspore::SWISH);
  return activation_cnode;
}
}  // namespace mindspore::opt
