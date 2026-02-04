/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/inplace_input_replace.h"

#include <unordered_map>
#include <string>
#include <vector>
#include <utility>
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "frontend/optimizer/irpass/view_inplace_utils.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
bool IsInplaceCNode(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad)) {
    return true;
  }
  auto prim = GetCNodePrimitive(node);
  return prim != nullptr && prim->inplace_prim();
}

bool HasIOMonadInput(const AnfNodePtr &node) {
  if (!irpass::IsCNode(node)) {
    return false;
  }
  const auto &cnode_inputs = node->cast<CNodePtr>()->inputs();
  return std::any_of(cnode_inputs.begin(), cnode_inputs.end(),
                     [](const AnfNodePtr &input) { return IsValueNode<IOMonad>(input); });
}

AnfNodePtr FindNodeUserWithIOMonad(const mindspore::CompactSet<std::pair<AnfNodePtr, int>> &node_users) {
  AnfNodePtr node_user_with_io_monad = nullptr;
  bool found = std::any_of(node_users.begin(), node_users.end(),
                           [&node_user_with_io_monad](const std::pair<AnfNodePtr, int> &node_user) {
                             bool has_io_monad_input = HasIOMonadInput(node_user.first);
                             if (has_io_monad_input) {
                               node_user_with_io_monad = node_user.first;
                             }
                             return has_io_monad_input;
                           });
  return found ? node_user_with_io_monad : nullptr;
}

void RecordInplaceNodes(const CNodePtr &cnode, std::unordered_map<AnfNodePtr, AnfNodePtr> *inplace_input) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &prim = GetCNodePrimitive(cnode);
  const auto &inputs = cnode->inputs();
  // Func call nodes and switch call nodes
  if (prim == nullptr) {
    if (IsPrimitiveCNode(inputs[kIndex0], prim::kPrimSwitch)) {
      auto switch_cnode = inputs[kIndex0]->cast<CNodePtr>();
      auto true_fg = GetValueNode<FuncGraphPtr>(switch_cnode->input(kIndex2));
      auto false_fg = GetValueNode<FuncGraphPtr>(switch_cnode->input(kIndex3));
      auto true_index = IsFuncOutputSameWithParamNode(true_fg);
      if (true_index == -1) {
        return;
      }
      auto false_index = IsFuncOutputSameWithParamNode(false_fg);
      if (true_index == false_index) {
        (*inplace_input)[cnode->input(true_index + 1)] = cnode;
        MS_LOG(INFO) << "Record inplace switch call cnode as inplace node: " << cnode->DebugString();
      }
    } else {
      auto fg = GetValueNode<FuncGraphPtr>(inputs[kIndex0]);
      if (auto index = IsFuncOutputSameWithParamNode(fg); index != -1) {
        (*inplace_input)[cnode->input(index + 1)] = cnode;
        MS_LOG(INFO) << "Record inplace call cnode as inplace node: " << cnode->DebugString();
      }
    }
    return;
  }

  // Inplace op nodes
  if (prim->inplace_prim()) {
    const auto &indexes = prim->inplace_input_indexes();
    if (indexes.size() != 1) {
      return;
    }
    (*inplace_input)[cnode->input(LongToSize(indexes[0] + 1))] = cnode;
    MS_LOG(INFO) << "Record cnode as inplace node: " << cnode->DebugString();
    return;
  }

  if (IsPrimitiveCNode(cnode, prim::kPrimVirtualViewGrad)) {
    (*inplace_input)[cnode->input(1)] = cnode;
    MS_LOG(INFO) << "Record VirtualViewGrad cnode as inplace node: " << cnode->DebugString();
    return;
  }
}

bool IsParamAssignCNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimAssign)) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() > kIndex1,
                             "Input size should be larger than 1, cnode: " + cnode->DebugString());
  const auto &assigned_node = cnode->input(kIndex1);
  const auto abs = assigned_node->abstract();
  if (abs == nullptr || !abs->isa<abstract::AbstractRefTensor>()) {
    return false;
  }
  auto abs_ref = abs->cast<abstract::AbstractRefPtr>();
  return abs_ref->is_parameter();
}

/**
 * \brief Change inplace input of cnode in func_graph.
 *
 * \example
 * Change from:
 *   %0 = InplaceOp(param_x, param_y)
 *   %1 = UpdataState(U, %0)
 *   %2 = Depend(param_x, %1)
 * To:
 *   %0 = InplaceOp(param_x, param_y)
 *   %1 = UpdataState(U, %0)
 *   %2 = Depend(%0, %1)
 *
 * \param[in] func_graph func graph.
 **/
void ChangeInplaceInputInner(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::unordered_map<AnfNodePtr, AnfNodePtr> inplace_input;
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users_map = manager->node_users();
  auto output_node = func_graph->output();
  for (auto node : TopoSort(output_node)) {
    if (!irpass::IsCNode(node) || IsPrimitiveCNode(node, prim::kPrimVirtualAssignAdd) ||
        node->func_graph() != func_graph) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    // If cnode has users with io_monad, do not do replacement
    // cnode1 = Load(param, u)
    // Print(str, cnode1, io)
    if (auto it = node_users_map.find(cnode); it != node_users_map.end()) {
      auto &node_users = it->second;
      auto node_user_with_io_monad = FindNodeUserWithIOMonad(node_users);
      if (node_user_with_io_monad != nullptr) {
        MS_LOG(INFO) << "CNode has users with io_monad, no need do replacement, cnode: " << cnode->DebugString()
                     << " , user cnode: " << node_user_with_io_monad->DebugString();
        continue;
      }
    }

    if (!IsParamAssignCNode(node)) {
      ReplaceInplaceNodeForCNode(cnode, inplace_input, manager, func_graph, true);
    }

    // Record nodes need to be replaced later
    RecordInplaceNodes(cnode, &inplace_input);
  }

  // Reprocess return node separately, avoid leaving any isolated nodes unreplaced
  if (!IsPrimitiveCNode(output_node, prim::kPrimDepend)) {
    return;
  }
  AnfNodePtr real_output = output_node;
  while (IsPrimitiveCNode(real_output, prim::kPrimDepend)) {
    real_output = real_output->cast<CNodePtr>()->input(kIndex1);
  }
  // real_output = {prim::kPrimMakeTuple, inplace_input, ...}
  // Isolated inplace nodes
  // Return {prim::kPrimDepend, real_output, ...}
  auto real_output_cnode = real_output->cast<CNodePtr>();
  if (IsPrimitiveCNode(real_output_cnode, prim::kPrimMakeTuple)) {
    ReplaceInplaceNodeForCNode(real_output_cnode, inplace_input, manager, func_graph);
  }
  return;
}
}  // namespace

bool DoInplaceInputReplace(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple);
  bool exist_inplace_nodes = std::any_of(all_nodes.begin(), all_nodes.end(), IsInplaceCNode);
  if (!exist_inplace_nodes) {
    return false;
  }
  auto manager = func_graph->manager();
  // Do inplace input replace for func_graph and sub_graphs
  ChangeInplaceInputInner(func_graph, manager);
  auto sub_graphs = func_graph->func_graphs_used_total();
  for (const auto &sub_graph : sub_graphs) {
    ChangeInplaceInputInner(sub_graph, manager);
  }

  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
