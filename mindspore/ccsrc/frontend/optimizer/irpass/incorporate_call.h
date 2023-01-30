/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_CALL_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_CALL_H_

#include <vector>
#include <algorithm>
#include <utility>
#include <memory>

#include "utils/hash_map.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
class CallOutputTransform {
 public:
  CallOutputTransform() : cache_() {}
  ~CallOutputTransform() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &fg, size_t nargs, bool xs_first) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    auto key = std::make_pair(nargs, xs_first);
    if (cache.find(key) == cache.end()) {
      FuncGraphPtr new_fg = TransformableClone(fg, std::make_shared<TraceTransform>("call"));

      std::vector<AnfNodePtr> new_items;
      new_items.push_back(new_fg->output());
      if (xs_first) {
        for (size_t i = 0; i < nargs; i++) {
          new_items.push_back(new_fg->add_parameter());
        }
      } else {
        for (size_t i = 0; i < nargs; i++) {
          new_items.push_back(new_fg->InsertFrontParameter());
        }
      }
      new_fg->set_output(new_fg->NewCNode(new_items));

      cache[key] = new_fg;
    }
    return cache[key];
  }

 private:
  mindspore::HashMap<FuncGraphPtr, mindspore::HashMap<std::pair<size_t, bool>, FuncGraphPtr, PairHasher>> cache_;
};
}  // namespace internal

// {{G, Xs}, Ys}
class IncorporateCall : public AnfVisitor {
 public:
  IncorporateCall() : call_output_transform_() {}
  ~IncorporateCall() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (inputs[0] == nullptr || !inputs[0]->isa<CNode>()) {
      return nullptr;
    }

    AnfVisitor::Visit(inputs[0]);
    if (fg_ == nullptr) {
      return nullptr;
    }

    auto xs_size = Xs_.size();
    auto ys_size = inputs.size() - 1;
    bool xs_first = true;
    if ((xs_size > 0) && (Xs_[xs_size - 1]->abstract() != nullptr) &&
        (Xs_[xs_size - 1]->abstract()->isa<abstract::AbstractMonad>())) {
      xs_first = false;
    }
    auto new_fg = call_output_transform_(fg_, ys_size, xs_first);

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(new_fg));

    if (xs_first) {
      if (xs_size > 0) {
        (void)args.insert(args.cend(), Xs_.cbegin(), Xs_.cend());
      }
      if (ys_size > 0) {
        (void)args.insert(args.cend(), inputs.cbegin() + 1, inputs.cend());
      }
    } else {
      if (ys_size > 0) {
        (void)args.insert(args.cend(), inputs.cbegin() + 1, inputs.cend());
      }
      if (xs_size > 0) {
        (void)args.insert(args.cend(), Xs_.cbegin(), Xs_.cend());
      }
    }
    return MakeNewNode(node, args);
  }

  AnfNodePtr MakeNewNode(const AnfNodePtr &node, const std::vector<AnfNodePtr> &args) {
    auto new_node = node->func_graph()->NewCNode(args);
    new_node->set_abstract(node->abstract());
    // Check if the another only usage of {G, Xs} is UpdateState{s, {G, Xs}}, if yes, replace
    // UpdateState{s, {G, Xs}} with UpdateState{s, new_node};
    const auto &manager = fg_->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &node_users_map = manager->node_users();
    auto it = node_users_map.find(fg_call_cnode_);
    if (it != node_users_map.end()) {
      AnfNodePtr update_state_node = nullptr;
      auto &node_users = it->second;
      constexpr size_t users_size = 2;
      if (node_users.size() == users_size) {
        for (auto &node_user : node_users) {
          if (IsPrimitiveCNode(node_user.first, prim::kPrimUpdateState)) {
            update_state_node = node_user.first;
          }
        }
      }
      if (update_state_node != nullptr) {
        auto update_state_cnode = update_state_node->cast<CNodePtr>();
        // double check;
        const size_t attach_index = 2;
        if (update_state_cnode->input(attach_index) == fg_call_cnode_) {
          constexpr int recursive_level = 2;
          MS_LOG(DEBUG) << "Replace UpdateState node: " << update_state_cnode->DebugString(recursive_level)
                        << ", input 2 with: " << new_node->DebugString();
          manager->SetEdge(update_state_cnode, attach_index, new_node);
        }
      }
    }
    return new_node;
  }

  void Visit(const CNodePtr &cnode) override {
    // {G, Xs}
    if (cnode->size() < 1 || !IsValueNode<FuncGraph>(cnode->input(0))) {
      return;
    }

    auto &inputs = cnode->inputs();
    fg_ = GetValueNode<FuncGraphPtr>(inputs[0]);
    fg_call_cnode_ = cnode;
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(Xs_));
  }

  void Reset() {
    Xs_.clear();
    fg_ = nullptr;
    fg_call_cnode_ = nullptr;
  }

 private:
  FuncGraphPtr fg_;
  CNodePtr fg_call_cnode_{nullptr};
  std::vector<AnfNodePtr> Xs_{};
  internal::CallOutputTransform call_output_transform_;
};

// {{{prim::kPrimSwitch, X, G1, G2}, Xs}, Ys}
class IncorporateCallSwitch : public AnfVisitor {
 public:
  IncorporateCallSwitch() : call_output_transform_() {}
  ~IncorporateCallSwitch() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {{...}, Ys}
    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (inputs[0] == nullptr || !inputs[0]->isa<CNode>()) {
      return nullptr;
    }

    // {{...}, Xs}
    auto &inputs_x = inputs[0]->cast<CNodePtr>()->inputs();
    if (inputs_x[0] == nullptr || !inputs_x[0]->isa<CNode>()) {
      return nullptr;
    }

    // {prim::kPrimSwitch, X, G1, G2}
    AnfVisitor::Match(prim::kPrimSwitch, {IsNode, IsValueNode<FuncGraph>, IsValueNode<FuncGraph>})(inputs_x[0]);
    if (g2_ == nullptr) {
      return nullptr;
    }

    auto fg = node->func_graph();
    auto xs_size = inputs_x.size() - 1;
    auto ys_size = inputs.size() - 1;
    bool xs_first = true;
    if ((xs_size > 0) && (inputs_x[xs_size]->abstract() != nullptr) &&
        (inputs_x[xs_size]->abstract()->isa<abstract::AbstractMonad>())) {
      xs_first = false;
    }
    auto new_g1 = call_output_transform_(g1_, ys_size, xs_first);
    auto new_g2 = call_output_transform_(g2_, ys_size, xs_first);
    auto sw_node = fg->NewCNode({NewValueNode(prim::kPrimSwitch), x_, NewValueNode(new_g1), NewValueNode(new_g2)});

    std::vector<AnfNodePtr> args{sw_node};
    if (xs_first) {
      if (xs_size > 0) {
        (void)args.insert(args.cend(), inputs_x.cbegin() + 1, inputs_x.cend());
      }
      if (ys_size > 0) {
        (void)args.insert(args.cend(), inputs.cbegin() + 1, inputs.cend());
      }
    } else {
      if (ys_size > 0) {
        (void)args.insert(args.cend(), inputs.cbegin() + 1, inputs.cend());
      }
      if (xs_size > 0) {
        (void)args.insert(args.cend(), inputs_x.cbegin() + 1, inputs_x.cend());
      }
    }

    auto new_node = fg->NewCNode(args);
    new_node->set_abstract(node->abstract());
    return new_node;
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
  internal::CallOutputTransform call_output_transform_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_CALL_H_
