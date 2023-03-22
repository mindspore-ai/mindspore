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

#include "tools/optimizer/fusion/add_activation_fusion.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "ops/fusion/add_fusion.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "tools/converter/quantizer/quant_param_holder.h"

namespace mindspore::opt {
const BaseRef AddActivationFusion::DefinePattern() const {
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  auto is_act = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  auto act = VectorRef({is_act, is_add});
  return act;
}

const AnfNodePtr AddActivationFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto act_cnode = node->cast<CNodePtr>();
  if (act_cnode == nullptr) {
    MS_LOG(ERROR) << "node is not cnode";
    return nullptr;
  }

  std::set<int64_t> support_act_types{mindspore::RELU, mindspore::RELU6};
  if (!CheckPattern(func_graph, act_cnode, support_act_types)) {
    return nullptr;
  }

  auto add_cnode = act_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_cnode != nullptr, nullptr);
  auto act_prim = ops::GetOperator<mindspore::ops::Activation>(act_cnode->input(0));
  MS_CHECK_TRUE_RET(act_prim != nullptr, nullptr);
  auto type = act_prim->get_activation_type();

  auto add_prim = ops::GetOperator<ops::AddFusion>(add_cnode->input(0));
  MS_CHECK_TRUE_RET(add_prim != nullptr, nullptr);
  (void)add_prim->AddAttr(ops::kActivationType, api::MakeValue<int64_t>(static_cast<int64_t>(type)));

  // copy the output quant params of activation to add node
  auto add_primitive = GetValueNode<PrimitivePtr>(add_cnode->input(0));
  auto act_primitive = GetValueNode<PrimitivePtr>(act_cnode->input(0));
  MS_CHECK_TRUE_RET(add_primitive != nullptr, nullptr);
  MS_CHECK_TRUE_RET(act_primitive != nullptr, nullptr);
  auto act_quant_params_valueptr = act_primitive->GetAttr("quant_params");
  if (act_quant_params_valueptr != nullptr) {
    auto act_quant_param_holder = act_quant_params_valueptr->cast<lite::QuantParamHolderPtr>();
    if (act_quant_param_holder->IsOutputExistInited()) {
      auto quant_params = act_quant_param_holder->get_output_quant_params();
      auto add_quant_params_valueptr = add_primitive->GetAttr("quant_params");
      if (add_quant_params_valueptr != nullptr) {
        auto add_quant_params_holder = add_quant_params_valueptr->cast<lite::QuantParamHolderPtr>();
        add_quant_params_holder->set_output_quant_param(0, quant_params[0]);
      } else {
        auto add_quant_params_holder = std::make_shared<lite::QuantParamHolder>(0, 0);
        add_quant_params_holder->set_output_quant_param(0, quant_params[0]);
        add_primitive->AddAttr("quant_params", add_quant_params_holder);
      }
    }
  }
  return add_cnode;
}

bool AddActivationFusion::CheckPattern(const FuncGraphPtr &func_graph, const CNodePtr &act_cnode,
                                       const std::set<int64_t> support_act_types) const {
  MS_CHECK_TRUE_RET(act_cnode->input(1) != nullptr, false);
  if (!utils::isa<CNodePtr>(act_cnode->input(1))) {
    MS_LOG(ERROR) << "add is not cnode.";
    return false;
  }
  auto add_cnode = act_cnode->input(1)->cast<CNodePtr>();

  auto manage = Manage(func_graph);
  if (manage == nullptr) {
    MS_LOG(ERROR) << "manage is nullptr.";
    return false;
  }
  auto node_users = manage->node_users()[add_cnode];
  if (node_users.size() > 1) {
    MS_LOG(INFO) << "Add node has multiple outputs";
    return false;
  }

  auto add_prim = ops::GetOperator<ops::AddFusion>(add_cnode->input(0));
  if (add_prim == nullptr) {
    MS_LOG(ERROR) << "Add prim is nullptr.";
    return false;
  }
  auto add_act_type = add_prim->GetAttr(ops::kActivationType);
  if (add_act_type != nullptr && add_prim->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(INFO) << "Add has activation.";
    return false;
  }
  auto act_prim = ops::GetOperator<mindspore::ops::Activation>(act_cnode->input(0));
  if (act_prim == nullptr) {
    MS_LOG(ERROR) << "Activation prim is nullptr.";
    return false;
  }
  auto act_type = act_prim->GetAttr(ops::kActivationType);
  if (act_type == nullptr) {
    MS_LOG(INFO) << "Activation type attr is nullptr.";
    return false;
  }
  auto type = act_prim->get_activation_type();
  if (support_act_types.find(type) == support_act_types.end()) {
    return false;
  }
  return true;
}
}  // namespace mindspore::opt
