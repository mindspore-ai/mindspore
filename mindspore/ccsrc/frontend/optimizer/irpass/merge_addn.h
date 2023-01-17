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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_

#include <vector>
#include <algorithm>
#include <memory>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {PrimAddN, {prim::kPrimMakeTuple, {PrimAddN, {prim::kPrimMakeTuple, Xs}}, Ys}} ->
// {{PrimAddNClass}, {prim::kPrimMakeTuple, Xs, Ys}}
// {PrimAddN, {prim::kPrimMakeTuple, Ys, {PrimAddN, {prim::kPrimMakeTuple, Xs}}}} ->
// {{PrimAddNClass}, {prim::kPrimMakeTuple, Ys, Xs}}
class MergeAddN : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    Reset();
    mng_ = optimizer->manager();
    is_outer_ = true;
    AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(node);
    // do not hold this manager
    mng_ = nullptr;
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }
    addn_nodes_.push_back(node);

    auto cnode = node->cast<CNodePtr>();
    auto addn = NewValueNode(GetValueNode(cnode->input(0)));

    // {prim::kPrimMakeTuple, Xs, Ys}, {prim::kPrimMakeTuple, Ys, Xs}
    (void)args_.insert(args_.cbegin(), NewValueNode(prim::kPrimMakeTuple));
    auto fg = node->func_graph();
    auto make_node = fg->NewCNode(args_);

    auto new_node = fg->NewCNode({addn, make_node});
    UpdateDumpFlag(new_node);
    new_node->AddFusedDebugInfoList(addn_nodes_);
    return new_node;
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
      const auto &first_input = inputs.at(1);
      AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(first_input);
      if (is_match_) {
        if (!is_unique(first_input)) {
          is_match_ = false;
          return;
        }

        if (!IsStateEquivalent(cnode, first_input)) {
          is_match_ = false;
          return;
        }

        addn_nodes_.push_back(first_input);
        (void)Ys_.erase(Ys_.cbegin());
        (void)std::copy(Xs_.cbegin(), Xs_.cend(), std::back_inserter(args_));
        (void)std::copy(Ys_.cbegin(), Ys_.cend(), std::back_inserter(args_));
        return;
      }

      // {prim::kPrimMakeTuple, Ys, {PrimAddN, {prim::kPrimMakeTuple, Xs}}}
      const auto &last_input = inputs.back();
      AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(last_input);
      if (is_match_) {
        if (!is_unique(last_input)) {
          is_match_ = false;
          return;
        }

        if (!IsStateEquivalent(cnode, last_input)) {
          is_match_ = false;
          return;
        }

        addn_nodes_.push_back(last_input);
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
    auto &node_users = mng_->node_users();
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
    addn_nodes_.clear();
    is_inner_ = false;
    is_outer_ = false;
    is_match_ = false;
  }

  void UpdateDumpFlag(const AnfNodePtr &node) {
    if (node == nullptr) {
      return;
    }
    for (const auto &addn : addn_nodes_) {
      if (AnfUtils::GetDumpFlag(addn)) {
        AnfUtils::SetDumpFlag(node);
        return;
      }
    }
  }

 private:
  FuncGraphManagerPtr mng_{nullptr};
  std::vector<AnfNodePtr> Xs_{}, Ys_{}, args_{}, addn_nodes_{};
  bool is_inner_{false}, is_outer_{false}, is_match_{false};
};

// {PrimAddN, {kPrimMakeTuple, Xs}}
class AddNZeroFilter : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(node);

    if (filtered_Xs_.empty() || node->func_graph() == nullptr) {
      return nullptr;
    }

    // if only two node in filtered_nodes, {make_tuple, x}. return x.
    constexpr auto input_size = 2;
    if (filtered_Xs_.size() == input_size) {
      return filtered_Xs_[1];
    }

    // if only one node in filtered_nodes, all node is zerolike, return one of the input.
    if (filtered_Xs_.size() == 1 && Xs_.size() > 0) {
      return Xs_[0];
    }

    if (!has_zero_like_) {
      return nullptr;
    }

    auto cnode = node->cast<CNodePtr>();
    auto addn = NewValueNode(GetValueNode(cnode->input(0)));
    auto fg = node->func_graph();
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
      if (!IsPrimitiveCNode(x, prim::kPrimZerosLike)) {
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
  std::vector<AnfNodePtr> filtered_Xs_{}, Xs_{};
  bool has_zero_like_{false};
};

// {PrimAddN, {kPrimMakeTuple, Xs}}
class AddNCheckDump : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(node);

    // Only handle gradient addn.
    if (node->scope()->name().find("Gradients/") != 0) {
      return nullptr;
    }

    if (set_dump_) {
      AnfUtils::SetDumpFlag(node);
    }

    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (!IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      return;
    }
    if (cnode->size() < kSizeThree) {
      return;
    }

    // When all of inputs has dump flag, we need set dump flag for AddN.
    set_dump_ = true;
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto input = cnode->input(i);
      MS_EXCEPTION_IF_NULL(input);
      if (IsPrimitiveCNode(input, prim::kPrimTupleGetItem) || IsPrimitiveCNode(input, prim::kPrimDepend)) {
        input = input->cast<CNodePtr>()->input(kIndexOne);
      }
      if (!input->isa<CNode>() || !AnfUtils::GetDumpFlag(input)) {
        set_dump_ = false;
        break;
      }
    }
  }

  void Reset() { set_dump_ = false; }

 private:
  bool set_dump_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_
