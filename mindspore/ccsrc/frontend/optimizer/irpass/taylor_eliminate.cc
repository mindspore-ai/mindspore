/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "frontend/optimizer/irpass/taylor_eliminate.h"
#include <string>
#include <vector>
#include "ir/func_graph_cloner.h"
#include "pipeline/pynative/pynative_execute.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
// White list of ops with taylor rule.
const mindspore::HashSet<std::string> taylor_ops{
  prim::kPrimAdd->name(), prim::kPrimSub->name(), prim::kPrimRealDiv->name(),
  prim::kPrimMul->name(), prim::kPrimSin->name(), prim::kPrimCos->name(),
  prim::kPrimTan->name(), prim::kPrimExp->name(), prim::kPrimLog->name()};
// The ops below are excluded when considering taylor rules.
const mindspore::HashSet<std::string> taylor_exception_ops{prim::kPrimReturn->name(), prim::kPrimMakeTuple->name(),
                                                           prim::kPrimTupleGetItem->name(), prim::kPrimCast->name()};

// Cache list of primitive ops which have been replaced by taylor rule.
mindspore::HashMap<PrimitivePtr, FuncGraphPtr> taylor_ops_cache_;

FuncGraphPtr GetTaylorRule(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resources) {
  // Set a child scope named "grad'PrimitiveName'" for the taylor rule function,
  // and add "Gradients" to the front.
  static const std::string gradients_scope = "Gradients/";
  static const std::string grad_op_child_scope_prefix = "/grad";
  MS_EXCEPTION_IF_NULL(prim);
  auto scope = std::make_shared<Scope>(gradients_scope + ScopeManager::GetInstance().GetCurrentScope()->name() +
                                       grad_op_child_scope_prefix + prim->name());
  ScopeGuard scope_guard(scope);

  // Firstly we get taylor rule from mindir. If failed, parse the python function registered.
  FuncGraphPtr func_graph = nullptr;
  py::function taylor_fn;
  if (prim->is_base()) {
    taylor_fn = GetTaylorRuleFunction(prim->name());
  } else {
    taylor_fn = prim->cast<PrimitivePyPtr>()->GetTaylorRuleFunction();
    if (py::isinstance<py::none>(taylor_fn)) {
      taylor_fn = GetTaylorRuleFunction(prim->name());
    }
  }
  if (!taylor_fn || py::isinstance<py::none>(taylor_fn)) {
    MS_LOG(INFO) << "Fail to find taylor rule function for " << prim->name() << ". taylor_fn: " << py::str(taylor_fn);
    return nullptr;
  }
  func_graph = parse::ParsePythonCode(taylor_fn);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Fail to parse taylor rule function for " << prim->name() << ".";
    return nullptr;
  }
  auto taylor_rule_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP);
  if (taylor_rule_flag) {
    func_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  }
  pipeline::ResourceBasePtr res = (resources != nullptr) ? resources : std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(func_graph, res);
  return func_graph;
}

FuncGraphPtr GetTaylorPyObj(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resources) {
  auto fg = GetTaylorRule(prim, resources);
  return fg;
}

FuncGraphPtr GetTaylorPrimitive(const AnfNodePtr &node, const pipeline::ResourceBasePtr &resources) {
  auto prim_node = GetValueNode<PrimitivePtr>(node);
  MS_EXCEPTION_IF_NULL(prim_node);
  auto iter = taylor_ops_cache_.find(prim_node);
  if (iter != taylor_ops_cache_.end()) {
    return iter->second;
  }
  FuncGraphPtr primitive_taylor = GetTaylorPyObj(prim_node, resources);
  MS_EXCEPTION_IF_NULL(primitive_taylor);
  taylor_ops_cache_[prim_node] = primitive_taylor;
  return primitive_taylor;
}

FuncGraphPtr TaylorFunctor(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &resources) {
  const auto &value_nodes = func_graph->value_nodes();
  auto manager = resources->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);
  std::vector<AnfNodePtr> taylor_node_list;
  for (const auto &value_pair : value_nodes) {
    auto node = value_pair.first;
    MS_EXCEPTION_IF_NULL(node);
    if (IsValueNode<Primitive>(node)) {
      auto prim_node = GetValueNode<PrimitivePtr>(node);
      if (taylor_ops.count(prim_node->name()) > 0) {
        taylor_node_list.push_back(node);
      } else if (taylor_exception_ops.count(prim_node->name()) == 0) {
        MS_LOG(EXCEPTION) << "The operation " << prim_node->name()
                          << " is not supported in taylor higher order differentiation currently.";
      }
    }
  }
  for (size_t i = 0; i < taylor_node_list.size(); i++) {
    FuncGraphPtr taylor_node_graph = GetTaylorPrimitive(taylor_node_list[i], resources);
    MS_EXCEPTION_IF_NULL(taylor_node_graph);
    (void)manager->Replace(taylor_node_list[i], NewValueNode(taylor_node_graph));
  }
  taylor_ops_cache_.clear();
  MS_LOG(INFO) << "return replaced taylor node: " << func_graph->ToString() << " replace end.";
  return func_graph;
}

AnfNodePtr ExpandTaylor(const ValueNodePtr &vnode, const pipeline::ResourceBasePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  if (IsValueNode<FuncGraph>(vnode)) {
    ScopeGuard scope_guard(vnode->scope());
    auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(DEBUG) << "Funcgraph: " << func_graph->ToString() << " will expandTaylor now";
    auto newfg = TaylorFunctor(func_graph, resource);
    return NewValueNode(newfg);
  }
  return nullptr;
}
}  // namespace internal

bool ExpandTaylorPrim::operator()(const FuncGraphPtr &, const OptimizerPtr &optimizer) {
  // Search all taylor nodes.
  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &taylor_node : prim_nodes_) {
    auto taylor_fg_node = taylor_node->input(1);
    MS_EXCEPTION_IF_NULL(taylor_fg_node);
    auto taylor_fg = GetValueNode<FuncGraphPtr>(taylor_fg_node);
    if (taylor_fg == nullptr) {
      MS_LOG(EXCEPTION) << "Unexpected Taylor node, input func graph should not be null, node: "
                        << taylor_fg_node->ToString();
    }
    // Copy original forward graph in case of the influence of usage in other place.
    auto taylor_fg_copy = BasicClone(taylor_fg, true);
    manager->AddFuncGraph(taylor_fg_copy);
    auto taylor_fg_copy_node = NewValueNode(taylor_fg_copy);
    // Return expanded taylor graph.
    auto expanded_taylor = internal::ExpandTaylor(taylor_fg_copy_node->cast<ValueNodePtr>(), optimizer->resource());
    (void)manager->Replace(taylor_node, expanded_taylor);
    change = true;
  }
  return change;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
