/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pre_activate/common/optimizer.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <initializer_list>

#include "pre_activate/common/pass_manager.h"
#include "session/anf_runtime_algorithm.h"
#include "ir/manager.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr HandleSexpVector(const BaseRef &sexp, const BaseRef &graph, bool multigraph);

ValueNodePtr CreateValueNodeWithSexp(const BaseRef &sexp) {
  if (utils::isa<int>(sexp)) {
    return NewValueNode(utils::cast<int>(sexp));
  }
  if (utils::isa<float>(sexp)) {
    return NewValueNode(utils::cast<float>(sexp));
  }
  if (utils::isa<bool>(sexp)) {
    return NewValueNode(utils::cast<bool>(sexp));
  }
  if (utils::isa<ValuePtr>(sexp)) {
    return NewValueNode(utils::cast<ValuePtr>(sexp));
  }
  return nullptr;
}

CNodePtr CreateCNodeWithGraph(const std::vector<AnfNodePtr> &input_nodes, const BaseRef &graph) {
  if (utils::isa<FuncGraphPtr>(graph)) {
    return std::make_shared<CNode>(input_nodes, utils::cast<FuncGraphPtr>(graph));
  }
  if (utils::isa<VarPtr>(graph)) {
    return std::make_shared<CNode>(input_nodes, utils::cast<VarPtr>(graph));
  }
  return nullptr;
}

VarNodePtr CreateVarNodeWithSexp(const BaseRef &sexp, const BaseRef &graph) {
  if (utils::isa<VarPtr>(graph)) {
    MS_LOG(DEBUG) << "make VarPtr " + graph.ToString();
    return std::make_shared<VarNode>(utils::cast<VarPtr>(sexp), nullptr);
  }
  if (utils::isa<FuncGraphPtr>(graph)) {
    MS_LOG(DEBUG) << "VarNode, should input a Var in graph. It's GraphPtr: " + graph.ToString();
    return std::make_shared<VarNode>(utils::cast<VarPtr>(sexp), utils::cast<FuncGraphPtr>(graph));
  }
  MS_LOG(ERROR) << "VarNode, should input a Var in graph. It's " + graph.ToString();
  return nullptr;
}

AnfNodePtr SexpToNode(const BaseRef &sexp, const BaseRef &graph, bool multigraph = false) {
  MS_LOG(DEBUG) << "SexpToNode sexp: " + sexp.ToString() + ", graph " + graph.ToString();
  if (utils::isa<VectorRef>(sexp)) {
    return HandleSexpVector(sexp, graph, multigraph);
  }
  if (utils::isa<VarPtr>(sexp)) {
    return CreateVarNodeWithSexp(sexp, graph);
  }
  if (utils::isa<AnfNodePtr>(sexp)) {
    return utils::cast<AnfNodePtr>(sexp);
  }
  auto value_node = CreateValueNodeWithSexp(sexp);
  if (value_node == nullptr) {
    MS_LOG(EXCEPTION) << "sexp cannot converted. sexp: " + sexp.ToString();
  }
  return value_node;
}

AnfNodePtr HandleSexpVector(const BaseRef &sexp, const BaseRef &graph, bool multigraph) {
  MS_LOG(DEBUG) << "HandleSexpVector sexp: " + sexp.ToString() + ", graph " + graph.ToString();
  std::vector<AnfNodePtr> input_nodes;
  const auto &tuple = utils::cast<VectorRef>(sexp);
  if (multigraph && utils::isa<VarPtr>(graph)) {
    for (auto &x : tuple) {
      AnfNodePtr node = SexpToNode(x, std::make_shared<Var>("G"), true);
      input_nodes.push_back(node);
    }
    VarPtr var_ptr = utils::cast<VarPtr>(graph);
    return std::make_shared<CNode>(input_nodes, var_ptr);
  }

  for (auto &x : tuple) {
    AnfNodePtr node = SexpToNode(x, graph, multigraph);
    input_nodes.push_back(node);
  }
  return CreateCNodeWithGraph(input_nodes, graph);
}
}  // namespace

static bool AnfEqual(const BaseRef &a, const BaseRef &b) {
  if (utils::isa<AnfNodePtr>(a) && utils::isa<AnfNodePtr>(b)) {
    auto a_node = utils::cast<AnfNodePtr>(a);
    auto b_node = utils::cast<AnfNodePtr>(b);
    if (IsValueNode<Primitive>(a_node) && IsValueNode<Primitive>(b_node)) {
      auto a_value_node = a_node->cast<ValueNodePtr>();
      auto a_value = a_value_node->value();
      auto a_prim = a_value->cast<PrimitivePtr>();

      auto b_value_node = b_node->cast<ValueNodePtr>();
      auto b_value = b_value_node->value();
      auto b_prim = b_value->cast<PrimitivePtr>();

      return a_prim->name() == b_prim->name();
    } else if (a_node->isa<ValueNode>() && b_node->isa<ValueNode>()) {
      auto a_value_node_ptr = a_node->cast<ValueNodePtr>();
      if (a_value_node_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "cast value node ptr fail";
      }
      auto a_value_ptr = a_value_node_ptr->value();
      if (a_value_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "value ptr is nullptr";
      }

      auto b_value_node_ptr = b_node->cast<ValueNodePtr>();
      if (b_value_node_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "cast value node ptr fail";
      }
      auto b_value_ptr = b_value_node_ptr->value();
      if (b_value_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "value ptr is nullptr";
      }

      return (*a_value_ptr) == (*b_value_ptr);
    }
    MS_LOG(DEBUG) << "check AnfNodePtr equal";
  }
  if (utils::isa<FuncGraphPtr>(a) && utils::isa<FuncGraphPtr>(b)) {
    MS_LOG(DEBUG) << "check GraphPtr equal";
  }
  return a == b;
}

static bool CNodeTypeEqual(const BaseRef &a, const BaseRef &b) {
  // To matchCNode and Kernel's type
  if (utils::isa<CNode>(a) && utils::isa<CNode>(b)) {
    return true;
  }
  return a.type() == b.type();
}

PatternProcessPass::PatternProcessPass(const std::string &name, bool multigraph)
    : NodePass(name),
      multigraph_(multigraph),
      pattern_engine_(PatternEngine(std::make_shared<DefaultVisitor>(),
                                    std::function<bool(const BaseRef &, const BaseRef &)>(AnfEqual),
                                    std::function<bool(const BaseRef &, const BaseRef &)>(CNodeTypeEqual))) {}

const BaseRef PatternProcessPass::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return BaseRef({X});
}

void PatternProcessPass::Build() {
  VarPtr fg = std::make_shared<Var>("RootG");
  BaseRef pattern = std::move(DefinePattern());
  pattern_ = SexpToNode(pattern, fg, multigraph_);
}

AnfNodePtr PatternProcessPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (pattern_ == nullptr) {
    Build();
  }

  auto empty_equiv = std::make_shared<Equiv>();
  EquivPtr equiv = pattern_engine_.Match(pattern_, node, empty_equiv);
  if (equiv != nullptr && !equiv->empty()) {
    return Process(func_graph, node, equiv);
  }
  return nullptr;
}

void GraphOptimizer::AddPassManager(const PassManagerPtr &pass_manager) {
  if (pass_manager != nullptr) {
    pass_managers_.push_back(pass_manager);
  }
}

FuncGraphPtr GraphOptimizer::Optimize(const FuncGraphPtr &func_graph, bool run_only_once) {
  MS_EXCEPTION_IF_NULL(func_graph);
  run_only_once_ = (pass_managers_.size() == 1) ? true : run_only_once;
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, false);
    func_graph->set_manager(manager);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < pass_managers_.size(); ++i) {
      const PassManagerPtr &pm = pass_managers_[i];
      if (pm != nullptr && pm->Run(func_graph)) {
        changed = true;
      }
    }
    if (run_only_once_) {
      break;
    }
  }

  std::vector<FuncGraphPtr> func_graphs;
  func_graphs.push_back(func_graph);
  manager->KeepRoots(func_graphs);
  (void)TopoSort(func_graph->get_return());
  return func_graph;
}
}  // namespace opt
}  // namespace mindspore
