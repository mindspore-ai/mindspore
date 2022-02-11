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

#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "pipeline/pynative/pynative_execute.h"

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

bool IsSideEffectOp(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto effect_info = GetPrimEffectInfo(GetCNodePrimitive(node));
  return effect_info.memory || effect_info.io;
}

AnfNodePtr ExpandJ(const ValueNodePtr &vnode, const OptimizerPtr &optimizer) {
  AnfNodePtr expanded_node = nullptr;
  if (IsValueNode<FuncGraph>(vnode)) {
    ScopeGuard scope_guard(vnode->scope());
    auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(DEBUG) << "Funcgraph: " << func_graph->ToString() << " will expandJ now";
    auto newfg = ad::Grad(func_graph, optimizer);
    expanded_node = NewValueNode(newfg);
  } else if (IsValueNode<Primitive>(vnode)) {
    expanded_node = ExpandJPrimitive(vnode, optimizer->resource());
  } else {
    return nullptr;
  }
  optimizer->set_is_first_order_j(false);
  return expanded_node;
}
}  // namespace internal

bool ExpandJPrim::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  // Check whether need to eliminate forward cnodes in pynative mode.
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    auto pynative_exec = pynative::PynativeExecutor::GetInstance();
    auto grad_exec = pynative_exec->grad_executor();
    bool eliminate_forward = grad_exec->eliminate_forward();
    grad_exec->set_eliminate_forward(eliminate_forward && prim_nodes_.empty());
  }
  // Expand j nodes that don't have embed j nodes.
  bool change = false;
  auto manager = optimizer->manager();
  for (auto &j_node : prim_nodes_) {
    auto expanded_j = internal::ExpandJ(j_node->input(1)->cast<ValueNodePtr>(), optimizer);
    manager->Replace(j_node, expanded_j);
    if (j_node->func_graph()->has_flag(FUNC_GRAPH_FLAG_K_GRAPH)) {
      MS_LOG(DEBUG) << j_node->func_graph()->ToString() << " has FUNC_GRAPH_FLAG_K_GRAPH flag.";
      j_node->func_graph()->set_flag(FUNC_GRAPH_FLAG_K_GRAPH, false);
    }
    change = true;
  }
  return change;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
