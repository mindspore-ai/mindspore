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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/mul_activation_fusion.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "ops/fusion/mul_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
const BaseRef MulActivationFusion::DefinePattern() const {
  auto is_act = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_act != nullptr, {});
  auto is_concat = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_concat != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  VectorRef pattern_ref = VectorRef({is_act, is_concat, is_seq_var});
  return pattern_ref;
}

const AnfNodePtr MulActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (!CheckPrimitiveType(node, prim::kPrimActivation)) {
    MS_LOG(INFO) << "node is not activation node";
    return nullptr;
  }
  auto act_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(act_cnode != nullptr, nullptr);
  auto act_prim = ops::GetOperator<ops::Activation>(act_cnode->input(0));
  MS_CHECK_TRUE_RET(act_prim != nullptr, nullptr);
  if (IsQuantParameterNode(act_prim->GetPrim())) {
    MS_LOG(INFO) << "node is a quant-node";
    return nullptr;
  }
  if (act_prim->get_activation_type() != ActivationType::RELU && act_prim->get_activation_type() != RELU6) {
    MS_LOG(INFO) << "activation is not relu or relu6";
    return nullptr;
  }
  auto mul_node = act_cnode->input(1);
  MS_CHECK_TRUE_RET(mul_node != nullptr, nullptr);
  auto mul_cnode = mul_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul_cnode != nullptr, nullptr);
  if (IsMultiOutputTensors(func_graph, mul_cnode)) {
    MS_LOG(INFO) << "mul has multiple out-nodes";
    return nullptr;
  }
  auto mul_prim = ops::GetOperator<ops::MulFusion>(mul_cnode->input(0));
  MS_CHECK_TRUE_RET(mul_prim != nullptr, nullptr);
  if (IsQuantParameterNode(mul_prim->GetPrim())) {
    MS_LOG(INFO) << "node is a quant-node";
    return nullptr;
  }
  if (mul_prim->get_activation_type() != NO_ACTIVATION) {
    MS_LOG(INFO) << "Mul already has activaton fusion, fusion type: " << mul_prim->get_activation_type();
    return nullptr;
  }
  mul_prim->set_activation_type(act_prim->get_activation_type());

  // delete activation node
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);
  (void)manager->Replace(act_cnode, mul_node);
  return nullptr;
}
}  // namespace mindspore::opt
