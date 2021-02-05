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
AnfNodePtr TransformMergeBranches(const AnfNodePtr &true_output_node, const AnfNodePtr &false_output_node,
                                  const AbstractBasePtr &true_graph_output_abs,
                                  const AbstractBasePtr &false_graph_output_abs, const AnfNodePtr &cond,
                                  const FuncGraphPtr &func_graph);
}  // namespace internal

// {{prim::kPrimSwitch, X, G1, G2}, Xs}
class ConvertSwitchReplacement : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    PatternNode<AnfNodePtr> cond, true_br, false_br;

    auto ConvertSwitchLambda = [&node, &cond, &true_br, &false_br]() -> AnfNodePtr {
      auto g1_ = GetValueNode<FuncGraphPtr>(true_br.GetNode(node));
      auto g2_ = GetValueNode<FuncGraphPtr>(false_br.GetNode(node));
      auto x_ = cond.GetNode(node);

      // for switch replace method, only graphs without graph inside can be replaced
      for (auto &item : g1_->value_nodes()) {
        auto value_node = item.first;
        if (IsValueNode<FuncGraph>(value_node)) {
          return nullptr;
        }
      }

      for (auto &item : g2_->value_nodes()) {
        auto value_node = item.first;
        if (IsValueNode<FuncGraph>(value_node)) {
          return nullptr;
        }
      }

      auto true_output = g1_->output()->abstract();
      auto false_output = g2_->output()->abstract();
      auto trans_g1 = internal::TransformGraphCondTrueBranchNodes(g1_, x_);
      auto trans_g2 = internal::TransformGraphCondFalseBranchNodes(g2_, x_);

      std::vector<AnfNodePtr> params;
      auto cnode = node->cast<CNodePtr>();
      if (cnode && cnode->size() > 1) {
        // There are arguments for the call of switch result,
        // usually these are monad states added by auto-monad.
        for (size_t i = 1; i < cnode->size(); ++i) {
          params.push_back(cnode->inputs().at(i));
        }
      }
      auto fg = node->func_graph();
      auto cloned_g1 = InlineClone(trans_g1, fg, params);
      auto cloned_g2 = InlineClone(trans_g2, fg, params);
      auto nnode = internal::TransformMergeBranches(cloned_g1, cloned_g2, true_output, false_output, x_, fg);

      return nnode;
    };

    MATCH_REPLACE_LAMBDA_IF(
      node, PCNode(PPrimitive(prim::kPrimSwitch, cond, true_br, false_br)).MinExtraNodes(0), ConvertSwitchLambda,
      true_br.CheckFunc(IsValueNode<FuncGraph>, node) && false_br.CheckFunc(IsValueNode<FuncGraph>, node));

    return nullptr;
  }
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
