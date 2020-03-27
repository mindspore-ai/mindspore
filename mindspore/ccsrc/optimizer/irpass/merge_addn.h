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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MERGE_ADDN_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MERGE_ADDN_H_

#include <vector>
#include <algorithm>

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {PrimAddN, {prim::kPrimMakeTuple, {PrimAddN, {prim::kPrimMakeTuple, Xs}}, Ys}} ->
// {{PrimAddNClass}, {prim::kPrimMakeTuple, Xs, Ys}}
// {PrimAddN, {prim::kPrimMakeTuple, Ys, {PrimAddN, {prim::kPrimMakeTuple, Xs}}}} ->
// {{PrimAddNClass}, {prim::kPrimMakeTuple, Ys, Xs}}
class MergeAddN : public AnfVisitor {
 public:
  MergeAddN() : PrimAddN_(prim::GetPythonOps("AddN", "mindspore.ops.operations")) {}
  ~MergeAddN() override = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    Reset();
    optimizer_ = optimizer;
    is_outer_ = true;
    AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(node);
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto fg = node->func_graph();
    // {PrimAddNClass}
    auto addn_node = fg->NewCNode({NewValueNode(PrimAddN_)});

    // {prim::kPrimMakeTuple, Xs, Ys}, {prim::kPrimMakeTuple, Ys, Xs}
    (void)args_.insert(args_.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto make_node = fg->NewCNode(args_);

    return fg->NewCNode({addn_node, make_node});
  }

  void Visit(const CNodePtr &cnode) override {
    if (!IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      return;
    }

    auto &inputs = cnode->inputs();

    if (is_outer_) {
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(Ys_));

      is_outer_ = false;
      is_inner_ = true;

      // {prim::kPrimMakeTuple, {PrimAddN, {prim::kPrimMakeTuple, Xs}}, Ys}
      AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(inputs[1]);
      if (is_match_) {
        if (!is_unique(inputs[1])) {
          is_match_ = false;
          return;
        }
        (void)Ys_.erase(Ys_.begin());
        (void)std::copy(Xs_.begin(), Xs_.end(), std::back_inserter(args_));
        (void)std::copy(Ys_.begin(), Ys_.end(), std::back_inserter(args_));
        return;
      }

      // {prim::kPrimMakeTuple, Ys, {PrimAddN, {prim::kPrimMakeTuple, Xs}}}
      AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(inputs.back());
      if (is_match_) {
        if (!is_unique(inputs.back())) {
          is_match_ = false;
          return;
        }
        Ys_.pop_back();
        (void)std::copy(Ys_.begin(), Ys_.end(), std::back_inserter(args_));
        (void)std::copy(Xs_.begin(), Xs_.end(), std::back_inserter(args_));
        return;
      }

      return;
    }

    if (is_inner_) {
      is_match_ = true;
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(Xs_));
    }
  }

  bool is_unique(const AnfNodePtr &node) {
    auto mng = optimizer_->resource()->manager();
    auto &node_users = mng->node_users();
    if (node_users.find(node) == node_users.end()) {
      return false;
    }

    size_t n_use = node_users[node].size();
    return n_use == 1;
  }

  void Reset() {
    Xs_.clear();
    Ys_.clear();
    args_.clear();
    is_inner_ = false;
    is_outer_ = false;
    is_match_ = false;
  }

 private:
  ValuePtr PrimAddN_;
  OptimizerPtr optimizer_{nullptr};
  std::vector<AnfNodePtr> Xs_{}, Ys_{}, args_{};
  bool is_inner_{false}, is_outer_{false}, is_match_{false};
};

// {PrimAddN, {kPrimMakeTuple, Xs}}
class AddNZeroFilter : public AnfVisitor {
 public:
  AddNZeroFilter() : PrimAddN_(prim::GetPythonOps("AddN", "mindspore.ops.operations")) {}
  ~AddNZeroFilter() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(node);

    if (filtered_Xs_.empty() || node->func_graph() == nullptr) {
      return nullptr;
    }

    // if only two node in filtered_nodes, {make_tuple, x}. return x.
    if (filtered_Xs_.size() == 2) {
      return filtered_Xs_[1];
    }

    // if only one node in filtered_nodes, all node is zerolike, return one of the input.
    if (filtered_Xs_.size() == 1 && Xs_.size() > 0) {
      return Xs_[0];
    }

    if (!has_zero_like_) {
      return nullptr;
    }

    auto fg = node->func_graph();
    auto addn = fg->NewCNode({NewValueNode(PrimAddN_)});
    auto make_tuple = fg->NewCNode(filtered_Xs_);
    return fg->NewCNode({addn, make_tuple});
  }

  void Visit(const CNodePtr &cnode) override {
    if (!IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      return;
    }

    auto &inputs = cnode->inputs();
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(Xs_));

    // {kPrimMakeTuple, X1, X2, ...}
    filtered_Xs_.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (auto &x : Xs_) {
      if (!IsPrimitiveCNode(x, prim::kPrimZerosLikeTensor)) {
        filtered_Xs_.push_back(x);
      } else {
        has_zero_like_ = true;
      }
    }
  }

  void Reset() {
    Xs_.clear();
    filtered_Xs_.clear();
    has_zero_like_ = false;
  }

 private:
  ValuePtr PrimAddN_;
  std::vector<AnfNodePtr> filtered_Xs_{}, Xs_{};
  bool has_zero_like_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_MERGE_ADDN_H_
