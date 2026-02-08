/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>

#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "ir/func_graph_cloner.h"
#include "include/utils/pynative/adapter.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
AnfNodePtr ExpandJPrimitive(const ValueNodePtr &vnode, const pipeline::ResourceBasePtr &resource) {
  ScopeGuard scope_guard(vnode->scope());
  auto newg = ad::Kprim(vnode, resource);
  if (newg != nullptr) {
    return NewValueNode(newg);
  }
  // when find in J failed, try in Jmeta
  auto prim = GetValueNode<PrimitivePtr>(vnode);
  MetaFuncGraphPtr meta = ad::Kmeta(prim, resource, vnode);
  if (meta != nullptr) {
    return NewValueNode(meta);
  }
  return nullptr;
}

AnfNodePtrList ExpandMultiJ(const FuncGraphVector &func_graphs, const OptimizerPtr &optimizer,
                            const std::vector<bool> &is_view_inplace) {
  AnfNodePtrList expanded_nodes;
  auto new_func_graphs = ad::GradMultiFuncGraph(func_graphs, optimizer, is_view_inplace, true, true);
  (void)std::transform(new_func_graphs.cbegin(), new_func_graphs.cend(), std::back_inserter(expanded_nodes),
                       [](const FuncGraphPtr &new_func_graph) {
                         MS_EXCEPTION_IF_NULL(new_func_graph);
                         return NewValueNode(new_func_graph);
                       });
  return expanded_nodes;
}

void AddHighOrderUnsupportAttr(const FuncGraphPtr &need_grad_fg, const AnfNodePtr &j_node,
                               const FuncGraphManagerPtr &manager) {
  if (!need_grad_fg->has_attr(FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON)) {
    return;
  }
  const auto &node_users = manager->node_users();
  auto iter = node_users.find(j_node);
  if (iter == node_users.end()) {
    return;
  }
  // Set attr for j_node users' func_graph
  for (auto &user_pair : iter->second) {
    auto user_node = user_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_node);
    auto caller_func_graph = user_node->func_graph();
    MS_EXCEPTION_IF_NULL(caller_func_graph);
    auto reason = need_grad_fg->get_attr(FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON);
    caller_func_graph->set_attr(FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON, reason);
  }
}
}  // namespace internal

void ExpandJPrim::CloneUsedPrimalGraph(const FuncGraphManagerPtr &manager, FuncGraphVector *func_graphs) const {
  MS_EXCEPTION_IF_NULL(func_graphs);
  size_t func_graphs_size = func_graphs->size();
  for (size_t i = 0; i < func_graphs_size; ++i) {
    const auto &used_total = (*func_graphs)[i]->func_graphs_used_total();
    for (size_t j = 0; j < func_graphs_size; ++j) {
      auto fg_j = (*func_graphs)[j];
      if (j == i || !used_total.contains(fg_j)) {
        continue;
      }
      auto new_fg = BasicClone(fg_j);
      for (auto &j_node : prim_nodes_) {
        auto j_node_fg = GetValueNode<FuncGraphPtr>(j_node->input(1));
        if (j_node_fg == nullptr || j_node_fg != fg_j) {
          continue;
        }
        (void)manager->Replace(j_node->input(1), NewValueNode(new_fg));
      }
      (*func_graphs)[j] = new_fg;
    }
  }
}

bool ExpandJPrim::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  // Expand j nodes that don't have embed j nodes.
  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  FuncGraphVector func_graphs;
  HashMap<AnfNodePtr, size_t> j_node_to_index_map;
  size_t index = 0;
  auto set_k_graph_flag = [](const FuncGraphPtr &func_graph) {
    if (func_graph->has_flag(FUNC_GRAPH_FLAG_K_GRAPH)) {
      MS_LOG(DEBUG) << func_graph->ToString() << " has FUNC_GRAPH_FLAG_K_GRAPH flag.";
      func_graph->set_flag(FUNC_GRAPH_FLAG_K_GRAPH, false);
    }
  };
  std::vector<bool> is_view_inplace;
  for (auto &j_node : prim_nodes_) {
    const auto &j_node_inp1 = j_node->input(1);
    is_view_inplace.push_back(j_node->has_user_data("has_view_inplace_grad"));
    if (IsValueNode<FuncGraph>(j_node_inp1)) {
      auto cur_func_graph = GetValueNode<FuncGraphPtr>(j_node_inp1);
      func_graphs.push_back(cur_func_graph);
      MS_LOG(DEBUG) << "FuncGraph: " << cur_func_graph->ToString() << " will expandJ now";
      j_node_to_index_map[j_node] = index++;
      internal::AddHighOrderUnsupportAttr(cur_func_graph, j_node, manager);
    } else if (IsValueNode<Primitive>(j_node_inp1)) {
      auto expanded_j = internal::ExpandJPrimitive(j_node_inp1->cast<ValueNodePtr>(), optimizer->resource());
      (void)manager->Replace(j_node, expanded_j);
      set_k_graph_flag(j_node->func_graph());
      change = true;
    }
  }
  CloneUsedPrimalGraph(manager, &func_graphs);

  auto grad_func_graphs = internal::ExpandMultiJ(func_graphs, optimizer, is_view_inplace);
  for (const auto &j_node_index_iter : j_node_to_index_map) {
    const auto &j_node = j_node_index_iter.first;
    (void)manager->Replace(j_node, grad_func_graphs[j_node_index_iter.second]);
    set_k_graph_flag(j_node->func_graph());
    change = true;
  }
  optimizer->set_is_first_order_j(false);
  return change;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
