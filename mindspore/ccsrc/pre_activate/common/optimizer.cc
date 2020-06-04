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
PatternProcessPass::PatternProcessPass(const std::string &name, bool multigraph)
    : NodePass(name),
      multigraph_(multigraph),
      pattern_engine_(PatternEngine(std::make_shared<DefaultVisitor>(),
                                    std::function<bool(const BaseRef &, const BaseRef &)>(AnfEqual),
                                    std::function<bool(const BaseRef &, const BaseRef &)>(CNodeTypeEqual))),
      primitive_vars_(std::make_shared<PrimitiveVarMap>()) {}

const BaseRef PatternProcessPass::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return BaseRef({X});
}

void PatternProcessPass::Build() {
  VarPtr fg = std::make_shared<Var>("RootG");
  BaseRef pattern = std::move(DefinePattern());
  pattern_ = SexpToNode(pattern, fg, primitive_vars_.get(), multigraph_);
}

AnfNodePtr PatternProcessPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (pattern_ == nullptr) {
    Build();
  }

  auto empty_equiv = std::make_shared<Equiv>();
  MS_EXCEPTION_IF_NULL(primitive_vars_);
  EquivPtr equiv = pattern_engine_.Match(pattern_, node, *primitive_vars_, empty_equiv);
  if (equiv != nullptr && !equiv->empty()) {
    return Process(func_graph, node, equiv);
  }
  return nullptr;
}

bool MultipleOutputPatternProcessPass::MatchAnotherPattern(const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  VarPtr fg = std::make_shared<Var>("RootG");
  auto empty_equiv = std::make_shared<Equiv>();
  MS_EXCEPTION_IF_NULL(child_primitive_vars_);
  EquivPtr another_equiv =
    child_pattern_engine_.Match(SexpToNode(DefineAnotherPattern(), fg, child_primitive_vars_.get(), true), node,
                                *child_primitive_vars_, empty_equiv);
  if (another_equiv != nullptr && !another_equiv->empty()) {
    return IsShareNodes(equiv, another_equiv);
  }
  return false;
}

void GraphOptimizer::AddPassManager(const PassManagerPtr &pass_manager) {
  if (pass_manager != nullptr) {
    pass_managers_.push_back(pass_manager);
  }
}

FuncGraphPtr GraphOptimizer::Optimize(const FuncGraphPtr &func_graph, bool run_only_once) {
  MS_EXCEPTION_IF_NULL(func_graph);
  run_only_once_ = (pass_managers_.size() == 1) ? true : run_only_once;
  // Performance risk by creating new manager each time
  auto manager = Manage(func_graph, true);

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
