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
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
const BaseRef SigmoidMulFusion::DefinePattern() const {
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  auto is_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var != nullptr, {});
  auto activation_input = VectorRef({is_activation, is_var});
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  return VectorRef({is_mul, is_var, activation_input});
}

// x * sigmoid(x) ->swish(x)
const AnfNodePtr SigmoidMulFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto mul_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(mul_cnode)) {
    return nullptr;
  }
  auto activation_cnode = mul_cnode->input(kInputIndexTwo)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(activation_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(activation_cnode)) {
    return nullptr;
  }
  // activation must sigmoid
  auto activation_prim = GetValueNode<std::shared_ptr<mindspore::ops::Activation>>(activation_cnode->input(0));
  MS_CHECK_TRUE_RET(activation_prim != nullptr, nullptr);
  if (activation_prim == nullptr || (activation_prim->GetAttr(ops::kActivationType) != nullptr &&
                                     activation_prim->get_activation_type() != mindspore::SIGMOID)) {
    return nullptr;
  }
  activation_prim->set_activation_type(mindspore::SWISH);
  return activation_cnode;
}
}  // namespace mindspore::opt
