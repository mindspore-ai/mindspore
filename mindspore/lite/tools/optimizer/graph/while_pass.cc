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
#include "tools/optimizer/graph/while_pass.h"
#include <vector>
#include <memory>
#include "ops/switch.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_adapter.h"

namespace mindspore::opt {

ValueNodePtr WhilePass::GetSwitchAnfPrim() {
  auto switch_prim = std::make_shared<mindspore::ops::Switch>();
  ValueNodePtr partial_anf_prim = NewValueNode(switch_prim);
  return partial_anf_prim;
}

bool WhilePass::Run(const FuncGraphPtr &graph) {
  auto node_list = TopoSort(graph->get_return());
  static int count = 0;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimWhile)) {
      continue;
    }
    auto while_cnode = node->cast<CNodePtr>();
    MS_ASSERT(while_cnode != nullptr);
    if (while_cnode->inputs().size() < kWhileMinInputSize) {
      MS_LOG(ERROR) << "while input is not right.";
      return false;
    }

    // the order is fixed.
    auto cond_vnode = while_cnode->input(kWhileCondIndex);
    auto body_vnode = while_cnode->input(kWhileBodyIndex);
    auto cond_fg = GetValueNode<std::shared_ptr<FuncGraph>>(cond_vnode);
    auto body_fg = GetValueNode<std::shared_ptr<FuncGraph>>(body_vnode);
    if (cond_fg == nullptr || body_fg == nullptr) {
      MS_LOG(ERROR) << "Get value as func_graph failed.";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_FAILED);
      return false;
    }
    std::vector<AnfNodePtr> cond_partial_op_inputs{cond_vnode};
    std::vector<AnfNodePtr> body_partial_op_inputs{body_vnode};
    cond_partial_op_inputs.insert(cond_partial_op_inputs.end(), while_cnode->inputs().begin() + kWhileMinInputSize,
                                  while_cnode->inputs().end());
    body_partial_op_inputs.insert(body_partial_op_inputs.end(), while_cnode->inputs().begin() + kWhileMinInputSize,
                                  while_cnode->inputs().end());
    static int idx = 0;
    auto cond_partial_node = graph->NewCNode(cond_partial_op_inputs);
    cond_partial_node->set_fullname_with_scope("Partial-while-cond-" + std::to_string(idx));
    cond_partial_node->set_abstract(cond_fg->output()->abstract());
    auto body_partial_node = graph->NewCNode(body_partial_op_inputs);
    body_partial_node->set_fullname_with_scope("Partial-while-body-" + std::to_string(idx));
    idx++;

    // concat body_fg output to cond_fg input
    auto body_output = body_fg->output();
    auto body_output_cnode = utils::cast<CNodePtr>(body_output);
    auto prim = GetValueNode<PrimitiveCPtr>(body_output_cnode->input(0));
    if (prim == nullptr) {
      MS_LOG(ERROR) << "Get PrimitiveC of node:" << body_output_cnode->fullname_with_scope() << " failed.";
      return false;
    }

    // concat body to cond
    std::vector<AnfNodePtr> body_to_cond_inputs{cond_vnode};
    if (CheckPrimitiveType(body_output_cnode, kPrimMakeTuple)) {
      for (size_t i = 1; i < body_output_cnode->inputs().size(); ++i) {
        body_to_cond_inputs.emplace_back(body_output_cnode->input(i));
      }
    } else {
      body_to_cond_inputs.emplace_back(body_output_cnode);
    }

    // concat body to cond
    auto body_to_cond_cnode = body_fg->NewCNode(body_to_cond_inputs);
    body_to_cond_cnode->set_fullname_with_scope("Partial-while-body-to-cond");
    auto body_fg_manager = body_fg->manager();
    body_fg_manager->Replace(body_fg->output(), body_to_cond_cnode);
    body_fg->set_output(body_to_cond_cnode);
    body_partial_node->set_abstract(cond_fg->output()->abstract());

    // create switch cnode
    ValueNodePtr switch_anf_primitive = GetSwitchAnfPrim();
    if (switch_anf_primitive == nullptr) {
      MS_LOG(ERROR) << "GetSwitchAnfPrim failed.";
      return false;
    }

    // insert switch node
    std::vector<AnfNodePtr> switch_op_inputs = {switch_anf_primitive, cond_partial_node, body_partial_node};
    auto switch_cnode = graph->NewCNode(switch_op_inputs);
    switch_cnode->set_fullname_with_scope("Switch-" + std::to_string(count++));

    AbstractBasePtrList abstract_list;
    auto body_fg_output_cnode = utils::cast<CNodePtr>(body_fg->output());
    for (auto &cnode : body_fg_output_cnode->inputs()) {
      if (!utils::isa<CNodePtr>(cnode) && !utils::isa<ParameterPtr>(cnode)) {
        continue;
      }
      abstract_list.push_back(cnode->abstract());
    }
    switch_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));

    // create cond partial cnode
    auto manager = graph->manager();
    if (!manager->Replace(while_cnode, switch_cnode)) {
      MS_LOG(ERROR) << "replace node failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::opt
