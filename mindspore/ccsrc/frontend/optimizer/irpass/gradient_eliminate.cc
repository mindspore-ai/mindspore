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
    MS_EXCEPTION_IF_NULL(func_graph);
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

bool ExpandJPrim::operator()(const FuncGraphPtr &root, const OptimizerPtr &optimizer) {
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

  bool change = false;
  auto manager = optimizer->manager();
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      auto j_node = node->cast<CNodePtr>();
      // If graph also contains J(FuncGraph) or J(Primitive), then ignore this graph.
      // ExpandJ innermost graph or primitive first.
      if (internal::CheckIfEmbedJ(j_node)) {
        continue;
      }
      auto expanded_j = internal::ExpandJ(j_node->input(1)->cast<ValueNodePtr>(), optimizer->resource());
      manager->Replace(j_node, expanded_j);
      change = true;
    }
  }
  return change;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
