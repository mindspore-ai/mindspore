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

bool IsSideEffectOp(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto effect_info = GetPrimEffectInfo(GetCNodePrimitive(node));
  return effect_info.memory || effect_info.io;
}

void CheckSwitchWithSideEffect(const FuncGraphPtr &fg) {
  AnfNodePtr switch_node = nullptr;
  AnfNodePtr side_effect_node = nullptr;
  auto all_graphs = fg->func_graphs_used_total();
  all_graphs.add(fg);
  for (auto &child_fg : all_graphs) {
    for (const auto &node : child_fg->nodes()) {
      if (switch_node == nullptr && IsPrimitiveCNode(node, prim::kPrimSwitch)) {
        switch_node = node;
      }
      if (side_effect_node == nullptr && IsSideEffectOp(node)) {
        side_effect_node = node;
      }
      if (switch_node != nullptr && side_effect_node != nullptr) {
        MS_LOG(ERROR)
          << "Control flow with side effect op[" << GetCNodeFuncName(side_effect_node->cast<CNodePtr>())
          << "] in training situation is not supported and grads may be wrong. Please remove the control flow "
             "statement or the side effect op.\n"
          << " Side effect node:" << side_effect_node->DebugString();
        return;
      }
    }
  }
}

AnfNodePtr ExpandJ(const ValueNodePtr &vnode, const pipeline::ResourceBasePtr &resource) {
  if (IsValueNode<FuncGraph>(vnode)) {
    ScopeGuard scope_guard(vnode->scope());
    auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
    // If a control flow network has side effect ops inside, which is not supported now, a error will be raised to
    // alert wrong grads.
    CheckSwitchWithSideEffect(func_graph);
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

bool ExpandJPrim::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  // Search all j nodes.
  GetJPrim(func_graph);
  // Get j nodes that don't have embed j nodes.
  std::vector<CNodePtr> todo;
  // If graph also contains J(FuncGraph) or J(Primitive), then ignore this graph.
  // ExpandJ innermost graph or primitive first.
  std::copy_if(j_nodes_.begin(), j_nodes_.end(), std::back_inserter(todo),
               [](const CNodePtr &j_node) { return !internal::CheckIfEmbedJ(j_node); });
  // Check whether need to eliminate forward cnodes in pynative mode.
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    auto pynative_exec = pynative::PynativeExecutor::GetInstance();
    auto grad_exec = pynative_exec->grad_executor();
    bool eliminate_forward = grad_exec->eliminate_forward();
    grad_exec->set_eliminate_forward(eliminate_forward && todo.empty());
  }
  // Expand j nodes that don't have embed j nodes.
  bool change = false;
  auto manager = optimizer->manager();
  for (auto &j_node : todo) {
    auto expanded_j = internal::ExpandJ(j_node->input(1)->cast<ValueNodePtr>(), optimizer->resource());
    manager->Replace(j_node, expanded_j);
    change = true;
  }
  return change;
}

void ExpandJPrim::GetJPrim(const FuncGraphPtr &func_graph) {
  j_nodes_.clear();
  AnfNodePtr ret = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      j_nodes_.push_back(node->cast<CNodePtr>());
    }
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
