/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/if_pass.h"
#include <vector>
#include <memory>
#include "mindspore/lite/include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_adapter.h"
#include "ops/switch.h"

namespace mindspore::opt {

ValueNodePtr IfPass::GetSwitchAnfPrim() {
  auto switch_prim = std::make_shared<ops::Switch>();
  if (switch_prim == nullptr) {
    MS_LOG(ERROR) << "new prim failed.";
    return nullptr;
  }
  ValueNodePtr switch_anf_prim = NewValueNode(switch_prim);
  return switch_anf_prim;
}

void IfPass::ReplaceInput(const std::vector<AnfNodePtr> &node_list, AnfNodePtr new_input_cnode, std::string para_name) {
  for (auto &node : node_list) {
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = utils::cast<CNodePtr>(node);
      for (size_t k = 0; k < cnode->inputs().size(); k++) {
        if (!utils::isa<ParameterPtr>(cnode->input(k))) {
          continue;
        }
        auto para_input = utils::cast<ParameterPtr>(cnode->input(k));
        if (para_input->name() == para_name) {
          cnode->set_input(k, new_input_cnode);
        }
      }
    }
  }
}

bool IfPass::Run(const FuncGraphPtr &graph) {
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimIf)) {
      continue;
    }
    auto if_cnode = node->cast<CNodePtr>();
    MS_ASSERT(if_cnode != nullptr);
    if (if_cnode->inputs().size() < kIfMinInputSize) {
      MS_LOG(ERROR) << "if input is not right.";
      return false;
    }

    // the order is fixed.
    auto then_vnode = if_cnode->input(kIfThenIndex);
    auto else_vnode = if_cnode->input(kIfElseIndex);
    auto cond_vnode = if_cnode->input(kIfCondIndex);

    // else_vnode->cast<ValueNodePtr>()->set_value()
    auto then_fg = GetValueNode<std::shared_ptr<FuncGraph>>(then_vnode);
    auto else_fg = GetValueNode<std::shared_ptr<FuncGraph>>(else_vnode);

    if (then_fg == nullptr || else_fg == nullptr) {
      MS_LOG(ERROR) << "Get value as func_graph failed.";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_FAILED);
      return false;
    }

    // create then partial cnode
    std::vector<AnfNodePtr> then_partial_op_inputs{then_vnode};

    // create else partial cnode
    std::vector<AnfNodePtr> else_partial_op_inputs{else_vnode};

    // add if op input to then_cnode and else_cnode
    then_partial_op_inputs.insert(then_partial_op_inputs.end(), if_cnode->inputs().begin() + kIfMinInputSize,
                                  if_cnode->inputs().end());
    else_partial_op_inputs.insert(else_partial_op_inputs.end(), if_cnode->inputs().begin() + kIfMinInputSize,
                                  if_cnode->inputs().end());

    auto then_partial_node = graph->NewCNode(then_partial_op_inputs);
    then_partial_node->set_fullname_with_scope(node->fullname_with_scope() + "-partial-if-then");
    then_partial_node->set_abstract(then_fg->output()->abstract());

    auto else_partial_node = graph->NewCNode(else_partial_op_inputs);
    else_partial_node->set_fullname_with_scope(node->fullname_with_scope() + "-partial-if-else");

    // create switch cnode
    ValueNodePtr switch_anf_primitive = GetSwitchAnfPrim();
    if (switch_anf_primitive == nullptr) {
      MS_LOG(ERROR) << "GetSwitchAnfPrim failed.";
      return false;
    }

    // insert switch node
    std::vector<AnfNodePtr> switch_op_inputs = {switch_anf_primitive, then_partial_node, else_partial_node, cond_vnode};
    switch_op_inputs.insert(switch_op_inputs.end(), if_cnode->inputs().begin() + kIfMinInputSize,
                            if_cnode->inputs().end());
    auto switch_cnode = graph->NewCNode(switch_op_inputs);
    switch_cnode->set_fullname_with_scope(node->fullname_with_scope() + "-Switch");
    switch_cnode->set_abstract(if_cnode->abstract());

    // create then partial cnode
    auto manager = graph->manager();
    auto node_users = manager->node_users()[if_cnode];
    for (auto &node_user : node_users) {
      manager->SetEdge(node_user.first, node_user.second, switch_cnode);
    }
  }

  return true;
}
}  // namespace mindspore::opt
