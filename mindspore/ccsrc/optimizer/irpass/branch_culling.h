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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_BRANCH_CULLING_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_BRANCH_CULLING_H_

#include <vector>
#include <algorithm>

#include "optimizer/optimizer.h"
#include "optimizer/irpass.h"
#include "ir/visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimSwitch, true, X, Y}
// {prim::kPrimSwitch, false, X, Y}
class SwitchSimplify : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    auto getx = [this](const AnfNodePtr &node) -> bool {
      this->x_ = node;
      return true;
    };
    auto gety = [this](const AnfNodePtr &node) -> bool {
      this->y_ = node;
      return true;
    };
    AnfVisitor::Match(prim::kPrimSwitch, {IsValueNode<BoolImm>, getx, gety})(node);

    // simplify the switch
    if (is_match_) {
      if (cond_) {
        return x_;
      }
      return y_;
    }

    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (!is_match_ && IsValueNode<BoolImm>(node)) {
      cond_ = GetValue<bool>(GetValueNode(node));
      is_match_ = true;
    }
  }

  void Reset() {
    x_ = nullptr;
    y_ = nullptr;
    cond_ = false;
    is_match_ = false;
  }

 private:
  bool is_match_{false}, cond_{false};
  AnfNodePtr x_{nullptr}, y_{nullptr};
};

// {prim::kPrimTupleGetItem, {prim::kPrimSwith, X0, X1, X2}, C} =>
// {prim::kPrimSwith, X0, {prim::kPrimTupleGetItem, X1, C}, {prim::kPrimTupleGetItem, X2, C}}
class FloatTupleGetItemSwitch : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);

    auto fg = node->func_graph();
    if (Xs_.empty() || c_ == nullptr || fg == nullptr) {
      return nullptr;
    }

    auto true_node = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), Xs_[1], c_});
    auto false_node = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), Xs_[2], c_});

    return fg->NewCNode({NewValueNode(prim::kPrimSwitch), Xs_[0], true_node, false_node});
  }

  void Visit(const CNodePtr &cnode) override {
    // {prim::kPrimSwith, X1, X2, X3}
    if (!IsPrimitiveCNode(cnode, prim::kPrimSwitch) || cnode->size() != 4) {
      return;
    }

    // copy X1, X2, X3
    auto &inputs = cnode->inputs();
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(Xs_));
  }

  void Visit(const ValueNodePtr &vnode) override { c_ = vnode; }

  void Reset() {
    Xs_.clear();
    c_ = nullptr;
  }

 private:
  AnfNodePtr c_{nullptr};
  std::vector<AnfNodePtr> Xs_{};
};

// {prim::kPrimEnvGetItem, {prim::kPrimSwitch, X1, X2, X3}, X4, X5} =>
// {prim::kPrimSwitch, X1, {prim::kPrimEnvGetItem, X2, X4, X5}, {prim::kPrimEnvGetItem, X3, X4, X5}}
class FloatEnvGetItemSwitch : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    AnfVisitor::Match(prim::kPrimEnvGetItem, {IsCNode, IsNode, IsNode})(node);
    if (!is_match_) {
      return nullptr;
    }

    // {prim::kPrimEnvGetItem, {...}, X4, X5}
    auto cnode = node->cast<CNodePtr>();
    auto sw_node = cnode->input(1)->cast<CNodePtr>();
    auto x4 = cnode->input(2);
    auto x5 = cnode->input(3);

    is_match_ = false;
    AnfVisitor::Match(prim::kPrimSwitch, {IsNode, IsNode, IsNode})(sw_node);
    if (!is_match_) {
      return nullptr;
    }

    // {prim::kPrimSwitch, X1, X2, X3}
    auto x1 = sw_node->input(1);
    auto x2 = sw_node->input(2);
    auto x3 = sw_node->input(3);

    auto fg = node->func_graph();
    if (fg == nullptr) {
      return nullptr;
    }

    auto true_node = fg->NewCNode({NewValueNode(prim::kPrimEnvGetItem), x2, x4, x5});
    auto false_node = fg->NewCNode({NewValueNode(prim::kPrimEnvGetItem), x3, x4, x5});

    return fg->NewCNode({NewValueNode(prim::kPrimSwitch), x1, true_node, false_node});
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
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
class ConvertSwitchReplacement : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    Reset();
    auto cnode = node->cast<CNodePtr>();
    if (cnode->size() < 1) {
      return nullptr;
    }

    // {prim::kPrimSwitch, X, G1, G2}
    AnfVisitor::Match(prim::kPrimSwitch, {IsNode, IsValueNode<FuncGraph>, IsValueNode<FuncGraph>})(cnode->input(0));
    if (g2_ == nullptr || g1_->output() == nullptr || g2_->output() == nullptr) {
      return nullptr;
    }
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
    auto fg = node->func_graph();
    auto cloned_g1 = InlineClone(trans_g1, fg, params);
    auto cloned_g2 = InlineClone(trans_g2, fg, params);
    auto nnode = internal::TransformMergeBranches(cloned_g1, cloned_g2, true_output, false_output, x_, fg);
    return nnode;
  }

  void Visit(const AnfNodePtr &node) override {
    if (x_ == nullptr) {
      x_ = node;
      return;
    }
    AnfVisitor::Visit(node);
  }

  void Visit(const ValueNodePtr &vnode) override {
    auto g = GetValueNode<FuncGraphPtr>(vnode);
    if (g1_ == nullptr) {
      g1_ = g;
    } else {
      g2_ = g;
    }
  }

  void Reset() {
    x_ = nullptr;
    g1_ = nullptr;
    g2_ = nullptr;
  }

 private:
  AnfNodePtr x_{nullptr};
  FuncGraphPtr g1_{nullptr}, g2_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_BRANCH_CULLING_H_
