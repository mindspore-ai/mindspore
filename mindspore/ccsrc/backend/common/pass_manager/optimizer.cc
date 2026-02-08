/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#include "include/backend/common/pass_manager/optimizer.h"
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "utils/anf_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "backend/common/pass_manager/visitor.h"

namespace mindspore {
namespace opt {
PatternPass::PatternPass(const std::string &name, bool multigraph)
    : NodePass(name),
      multigraph_(multigraph),
      pattern_engine_(PatternEngine(std::make_shared<Visitor>())),
      primitive_vars_(std::make_shared<PrimitiveVarMap>()),
      equiv_(std::make_shared<Equiv>()) {}

void PatternPass::GetOrigNodes() {
  MS_EXCEPTION_IF_NULL(equiv_);
  orig_nodes_.clear();
  for (auto &prim_var : *primitive_vars_) {
    auto equiv_iter = equiv_->find(prim_var.second);
    if (equiv_iter == equiv_->end()) {
      continue;
    }
    auto baseref = equiv_iter->second;
    if (!utils::isa<CNode>(baseref)) {
      continue;
    }
    auto node = utils::cast<AnfNodePtr>(baseref);
    if (AnfUtils::IsRealCNodeKernel(node)) {
      orig_nodes_.push_back(node);
    }
  }
}

bool PatternPass::CheckNodeStreamAndCoreAttrs(const FuncGraphPtr &func_graph) const {
  return CheckStreamAndCoreAttrWithOrigNodes(func_graph, orig_nodes_);
}

CNodePtr PatternPass::NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  return opt::NewCNode(inputs, fg, orig_nodes_);
}

CNodePtr PatternPass::NewCNode(const CNodePtr &cnode, const KernelGraphPtr &fg) const {
  MS_EXCEPTION_IF_NULL(fg);
  return opt::NewCNode(cnode, fg, orig_nodes_);
}

const BaseRef PatternProcessPass::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return BaseRef({X});
}

void PatternProcessPass::Build() {
  VarPtr fg = std::make_shared<Var>("RootG");
  pattern_ = SexpToNode(DefinePattern(), fg, primitive_vars_.get(), multigraph_);
}

AnfNodePtr PatternProcessPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (pattern_ == nullptr) {
    Build();
  }

  auto primitive = GetCNodePrimitive(pattern_);
  if (IsPrimitiveCNode(node, primitive)) {
    MS_EXCEPTION_IF_NULL(primitive_vars_);
    MS_EXCEPTION_IF_NULL(equiv_);
    equiv_->clear();
    EquivPtr equiv = pattern_engine_.Match(pattern_, node, *primitive_vars_, equiv_);
    if (equiv != nullptr && !equiv->empty()) {
      GetOrigNodes();
      if (CheckNodeStreamAndCoreAttrs(func_graph)) {
        return Process(func_graph, node, equiv);
      }
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
