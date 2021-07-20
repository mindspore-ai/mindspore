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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BRANCH_CULLING_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BRANCH_CULLING_H_

#include <vector>
#include <algorithm>

#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "ir/pattern_matcher.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimSwitch, true, X, Y}
// {prim::kPrimSwitch, false, X, Y}
class SwitchSimplify : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> cond, true_br, false_br;
    auto SwitchSimplLambda = [&node, &cond, &true_br, &false_br]() -> AnfNodePtr {
      auto cond_value_ = GetValue<bool>(GetValueNode(cond.GetNode(node)));
      if (cond_value_) {
        return true_br.GetNode(node);
      }
      return false_br.GetNode(node);
    };

    MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), SwitchSimplLambda,
                            cond.CheckFunc(IsValueNode<BoolImm>, node));

    return nullptr;
  }
};

// {prim::kPrimTupleGetItem, {prim::kPrimSwitch, X0, X1, X2}, C} =>
// {prim::kPrimSwitch, X0, {prim::kPrimTupleGetItem, X1, C}, {prim::kPrimTupleGetItem, X2, C}}
class FloatTupleGetItemSwitch : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> cond, true_br, false_br, x;
    MATCH_REPLACE_IF(node,
                     PPrimitive(prim::kPrimTupleGetItem, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), x),
                     PPrimitive(prim::kPrimSwitch, cond, PPrimitive(prim::kPrimTupleGetItem, true_br, x),
                                PPrimitive(prim::kPrimTupleGetItem, false_br, x)),
                     x.CheckFunc(IsVNode, node));
    return nullptr;
  }
};

// {prim::kPrimEnvGetItem, {prim::kPrimSwitch, X1, X2, X3}, X4, X5} =>
// {prim::kPrimSwitch, X1, {prim::kPrimEnvGetItem, X2, X4, X5}, {prim::kPrimEnvGetItem, X3, X4, X5}}
class FloatEnvGetItemSwitch : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> cond, true_br, false_br, x, x2;
    MATCH_REPLACE(node,
                  PPrimitive(prim::kPrimEnvGetItem, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), x, x2),
                  PPrimitive(prim::kPrimSwitch, cond, PPrimitive(prim::kPrimEnvGetItem, true_br, x, x2),
                             PPrimitive(prim::kPrimEnvGetItem, false_br, x, x2)));

    return nullptr;
  }
};

namespace internal {
FuncGraphPtr TransformGraphCondTrueBranchNodes(const FuncGraphPtr &graph, const AnfNodePtr &cond);
FuncGraphPtr TransformGraphCondFalseBranchNodes(const FuncGraphPtr &graph, const AnfNodePtr &cond);
// block_nodes[0]: condition node
// block_nodes[1]: true branch node
// block_nodes[2]: false branch node
// branch_output_abs[0]: true branch abstract
// branch_output_abs[1]: false branch abstract
AnfNodePtr TransformMergeBranches(const std::vector<AnfNodePtr> &block_nodes,
                                  const std::vector<AbstractBasePtr> &branch_output_abs,
                                  const FuncGraphPtr &func_graph);
}  // namespace internal

// {{prim::kPrimSwitch, X, G1, G2}, Xs}
class ConvertSwitchReplacement {
 public:
  ConvertSwitchReplacement() = default;
  virtual ~ConvertSwitchReplacement() = default;

  bool operator()(const FuncGraphPtr &root, const OptimizerPtr &optimizer) {
    AnfNodePtr ret = root->get_return();
    MS_EXCEPTION_IF_NULL(ret);
    std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

    bool change = false;
    for (auto &node : all_nodes) {
      if (CheckSwitchWrapNode(node)) {
        TransformSwitchBranchReplace(node);
        change = true;
      }
    }
    return change;
  }

 private:
  // Determine whether there are graphs inside the branch graph.
  bool CheckSwitchBranch(const AnfNodePtr &node);
  // Determine whether node matches {{prim::kPrimSwitch, X, G1, G2}, Xs}.
  bool CheckSwitchWrapNode(const AnfNodePtr &node);
  // Replace switch branch.
  void TransformSwitchBranchReplace(const AnfNodePtr &node);
};

// {prim::kPrimSwitch, {prim::kPrimDepend, ValueNode, X}, G1, G2} ->
// {prim::kPrimDepend, {prim::kPrimSwitch, ValueNode, G1, G2}, X}
class ExchangeSwitchDependValue : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }
    ScopePtr scope = node->cast<CNodePtr>()->scope();
    ScopeGuard scope_guard(scope);

    PatternNode<AnfNodePtr> cond, true_br, false_br, v, x;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSwitch, PPrimitive(prim::kPrimDepend, v, x), true_br, false_br),
                     PPrimitive(prim::kPrimDepend, PPrimitive(prim::kPrimSwitch, v, true_br, false_br), x),
                     IsVNode(v.GetNode(node)));
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BRANCH_CULLING_H_
