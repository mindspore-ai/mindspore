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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ADD_FORWARD_MONAD_DEPEND_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ADD_FORWARD_MONAD_DEPEND_H_

#include <vector>
#include "include/frontend/optimizer/optimizer.h"
#include "include/frontend/optimizer/ad/grad_interface.h"
#include "frontend/optimizer/utils.h"
#include "ir/func_graph.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
bool IsMemSideEffectNode(const AnfNodePtr &node) {
  auto prim = GetCNodePrimitive(node);
  if (prim == nullptr) {
    return false;
  }
  return prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_MEM);
}

void AddUMonadInput(const FuncGraphManagerPtr &manager, const FuncGraphPtr &bprop_graph, const AnfNodePtr &new_u_para) {
  auto fprop_graph = bprop_graph->parent();
  auto is_bprop_node = [&fprop_graph](const AnfNodePtr &node) {
    if (node->func_graph() == fprop_graph) {
      return EXCLUDE;
    }
    return FOLLOW;
  };
  auto all_nodes = TopoSort(bprop_graph->get_return(), SuccDeeperSimple, is_bprop_node);
  for (const auto &node : all_nodes) {
    if (!IsMemSideEffectNode(node)) {
      continue;
    }
    MS_LOG(DEBUG) << "Add u monad input for node " << node->DebugString();
    manager->AddEdge(node, new_u_para);
  }
}

void PropagateUMonadInput(const FuncGraphManagerPtr &manager, const FuncGraphPtr &bprop_graph,
                          const AbstractBasePtr &u_abs, bool add_u_input) {
  auto new_u_para = bprop_graph->add_parameter();
  if (new_u_para->debug_info() != nullptr) {
    new_u_para->debug_info()->set_name("forward_u");
  }
  new_u_para->set_abstract(u_abs);
  bprop_graph->set_flag(kFuncGraphFlagAddedForwardU, true);
  if (add_u_input) {
    AddUMonadInput(manager, bprop_graph, new_u_para);
  }
  std::vector<CNodePtr> side_effect_bprop_app_propagate_nodes;
  for (const auto &node : bprop_graph->nodes()) {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode == nullptr) {
      continue;
    }
    if (cnode->HasAttr(kAttrSideEffectBpropAppPropagate) || cnode->HasAttr(kAttrSideEffectBpropApp)) {
      (void)side_effect_bprop_app_propagate_nodes.emplace_back(cnode);
    }
  }
  if (side_effect_bprop_app_propagate_nodes.empty()) {
    return;
  }

  for (const auto &propagate_node : side_effect_bprop_app_propagate_nodes) {
    manager->AddEdge(propagate_node, new_u_para);
    auto bprop_getter_abs = dyn_cast<abstract::FuncGraphAbstractClosure>(propagate_node->input(0)->abstract());
    if (bprop_getter_abs == nullptr) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, propagate_node) << "The node " << propagate_node->input(0)->DebugString()
                                                           << " should have a FuncGraphAbstractClosure abstract.";
    }
    auto bprop_fg = bprop_getter_abs->func_graph();
    MS_EXCEPTION_IF_NULL(bprop_fg);
    if (bprop_fg->has_flag(kFuncGraphFlagAddedForwardU)) {
      continue;
    }
    PropagateUMonadInput(manager, bprop_fg, u_abs, propagate_node->HasAttr(kAttrSideEffectBpropApp));
  }
}
}  // namespace internal

/**
 * \brief Pass of add forward monad depend.
 *
 * \example
 * The origin pattern:
 * %0 = U
 * %1 = call fprop(x, y, %0)
 * %2 = get_item(%1, 1)
 * %3 = %2[@@bprop](dout)
 *
 * graph bprop(dout):
 *   %0 = side_effect_mem_op(dout)
 *
 * After the pass:
 * kLevelNone(no changes)
 * %0 = U
 * %1 = call fprop(x, y, %0)
 * %2 = get_item(%1, 1)
 * %3 = %2[@@bprop](dout)
 *
 * graph bprop(dout):
 *   %0 = side_effect_mem_op(dout)
 *
 * kLevelTop
 * %0 = U
 * %1 = call fprop(x, y, %0)
 * %2 = get_item(%1, 1)
 * %3 = %2[@@bprop](dout, %0)
 *
 * graph bprop(dout, u):
 *   %0 = side_effect_mem_op(dout, u)
 *
 * kLevelWhole
 * %0 = U
 * %1 = call fprop(x, y, %0)
 * %2 = UpdateState(U, %1)
 * %3 = get_item(%1, 1)
 * %4 = %3[@@bprop](dout, %2)
 *
 * graph bprop(dout, u):
 *   %0 = side_effect_mem_op(dout, u)
 *
 * \param[in] root The root func graph.
 * \param[in] opt The optimizer.
 *
 * \return True if root func graph changed, else false.
 **/
bool AddForwardMonadDepend(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  auto scheduler = parallel::ParallelContext::GetInstance()->pipeline_scheduler();
  if (scheduler == mindspore::parallel::ZBV) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<FuncGraphPtr> top_k_graphs;
  for (const auto &fg : root->func_graphs_used_total()) {
    MS_EXCEPTION_IF_NULL(fg);
    if (fg->has_attr(kAttrBpropAutoMonadLevel) && fg->has_flag(kAttrSideEffectBpropAppPropagate)) {
      (void)top_k_graphs.emplace_back(fg);
    }
  }

  bool changed = false;
  for (const auto &top_k_graph : top_k_graphs) {
    auto bprop_auto_monad_level = GetValue<int>(top_k_graph->get_attr(kAttrBpropAutoMonadLevel));
    top_k_graph->erase_flag(kAttrBpropAutoMonadLevel);
    if (bprop_auto_monad_level == ad::BpropAutoMonadLevel::kLevelNone) {
      break;
    }
    FuncGraphPtr bprop_graph = nullptr;
    AbstractBasePtr u_abs = nullptr;
    for (const auto &entry : top_k_graph->func_graph_cnodes_index()) {
      auto k_graph_caller = entry.first->first->cast<CNodePtr>();
      auto index = entry.first->second;
      // Get the real graph caller.
      if (index != 0) {
        continue;
      }
      // Get the monad input.
      auto umonad_input = k_graph_caller->input(k_graph_caller->size() - 1);
      if (!HasAbstractUMonad(umonad_input)) {
        continue;
      }
      // Only handle the fprop which has bprop getter.
      auto bprop_getter = GetBpropGetter(manager, k_graph_caller);
      if (bprop_getter == nullptr) {
        continue;
      }
      auto bprop_getter_abs = dyn_cast<abstract::FuncGraphAbstractClosure>(bprop_getter->abstract());
      if (bprop_getter_abs == nullptr) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, bprop_getter)
          << "The node " << bprop_getter->DebugString() << " should have a FuncGraphAbstractClosure abstract.";
      }
      if (bprop_graph == nullptr) {
        bprop_graph = bprop_getter_abs->func_graph();
      } else if (bprop_getter_abs->func_graph() != bprop_graph) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, bprop_getter)
          << "The bprop graphs are not same for the k graph: " << top_k_graph->ToString();
      }
      auto bprop_user = GetBpropCaller(manager, bprop_getter);
      if (bprop_user == nullptr) {
        continue;
      }

      auto update_state_to_depend = umonad_input;
      if (bprop_auto_monad_level == ad::BpropAutoMonadLevel::kLevelWhole) {
        std::vector<AnfNodePtr> new_update_state_inputs = {NewValueNode(prim::kPrimUpdateState), umonad_input,
                                                           k_graph_caller};
        update_state_to_depend = k_graph_caller->func_graph()->NewCNodeInOrder(new_update_state_inputs);
        update_state_to_depend->set_abstract(umonad_input->abstract());
      }
      manager->AddEdge(bprop_user, update_state_to_depend);
      changed = true;
      u_abs = umonad_input->abstract();
    }
    if (bprop_graph != nullptr && u_abs != nullptr) {
      internal::PropagateUMonadInput(manager, bprop_graph, u_abs, false);
    }
  }
  return changed;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ADD_FORWARD_MONAD_DEPEND_H_
