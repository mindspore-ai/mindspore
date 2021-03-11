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

#include "frontend/optimizer/irpass/gradient_eliminate.h"

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
  MetaFuncGraphPtr meta = ad::Kmeta(prim, resource);
  if (meta != nullptr) {
    return NewValueNode(meta);
  }
  return nullptr;
}

bool CheckIfEmbedJ(const CNodePtr &j_node) {
  auto &value_node = j_node->input(1);
  if (IsValueNode<Primitive>(value_node)) {
    return false;
  }
  auto func_graph = GetValueNode<FuncGraphPtr>(value_node);
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Unexpected j node:" << j_node->DebugString();
  }
  auto func_graph_manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(func_graph_manager);
  return func_graph_manager->func_graph_j_total(func_graph);
}

AnfNodePtr ExpandJ(const ValueNodePtr &vnode, const pipeline::ResourceBasePtr &resource) {
  if (IsValueNode<FuncGraph>(vnode)) {
    ScopeGuard scope_guard(vnode->scope());
    auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
    MS_LOG(DEBUG) << "Funcgraph: " << func_graph->ToString() << " will expandJ now";
    auto newfg = ad::Grad(func_graph, resource);
    return NewValueNode(newfg);
  }
  if (IsValueNode<Primitive>(vnode)) {
    return ExpandJPrimitive(vnode, resource);
  }
  return nullptr;
}
}  // namespace internal

bool ExpandJPrim::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  // Search all j nodes.
  GetJPrim(optimizer->resource()->manager());
  // Get j nodes that don't have embed j nodes.
  std::vector<CNodePtr> todo;
  // If graph also contains J(FuncGraph) or J(Primitive), then ignore this graph.
  // ExpandJ innermost graph or primitive first.
  std::copy_if(j_nodes_.begin(), j_nodes_.end(), std::back_inserter(todo),
               [](const CNodePtr &j_node) { return !internal::CheckIfEmbedJ(j_node); });
  // Expand j nodes that don't have embed j nodes.
  bool change = false;
  for (auto &j_node : todo) {
    auto expanded_j = internal::ExpandJ(j_node->input(1)->cast<ValueNodePtr>(), optimizer->resource());
    optimizer->resource()->manager()->Replace(j_node, expanded_j);
    change = true;
  }
  return change;
}

void ExpandJPrim::GetJPrim(const FuncGraphManagerPtr &manager) {
  j_nodes_.clear();
  for (auto &fg : manager->func_graphs()) {
    std::vector<AnfNodePtr> &&toposet = TopoSort(fg->get_return());
    for (const auto &node : toposet) {
      if (IsPrimitiveCNode(node, prim::kPrimJ)) {
        j_nodes_.push_back(node->cast<CNodePtr>());
      }
    }
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
