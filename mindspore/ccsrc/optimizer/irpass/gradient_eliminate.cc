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

#include "optimizer/irpass/gradient_eliminate.h"

#include <utility>

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

bool CheckIfEmbedJFuncGraph(const FuncGraphPtr func_graph) {
  // if func graph also contain J FuncGraph, then ignore this funcgraph. ExpandJ innermost graph first;
  auto func_graph_manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(func_graph_manager);
  return func_graph_manager->func_graph_j_total(func_graph);
}

AnfNodePtr ExpandJ(const ValueNodePtr &vnode, const pipeline::ResourceBasePtr &resource) {
  if (IsValueNode<FuncGraph>(vnode)) {
    ScopeGuard scope_guard(vnode->scope());

    auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
    MS_LOG(DEBUG) << "Node is ValueNodeGraph, graph: " << func_graph->ToString();

    // high_order_grad begin;
    // if graph also contain J Graph, then ignore this graph. ExpandJ innermost graph first;
    if (CheckIfEmbedJFuncGraph(func_graph)) {
      MS_LOG(DEBUG) << "Funcgraph: " << func_graph->ToString() << " contains J(funcgraph), will expandJ later";
      return nullptr;
    }
    // high_order_grad end;

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
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
