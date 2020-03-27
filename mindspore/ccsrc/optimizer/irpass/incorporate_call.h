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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INCORPORATE_CALL_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INCORPORATE_CALL_H_

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <memory>

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
class CallOutputTransform {
 public:
  CallOutputTransform() : cache_() {}
  ~CallOutputTransform() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &fg, size_t nargs) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    if (cache.find(nargs) == cache.end()) {
      FuncGraphPtr new_fg = TransformableClone(fg, std::make_shared<TraceTransform>("call"));

      std::vector<AnfNodePtr> new_items;
      new_items.push_back(new_fg->output());
      for (size_t i = 0; i < nargs; i++) {
        new_items.push_back(new_fg->add_parameter());
      }
      new_fg->set_output(new_fg->NewCNode(new_items));

      cache[nargs] = new_fg;
    }
    return cache[nargs];
  }

 private:
  std::unordered_map<FuncGraphPtr, std::unordered_map<size_t, FuncGraphPtr>> cache_;
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
    auto new_fg = call_output_transform_(fg_, ys_size);

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(new_fg));

    if (xs_size > 0) {
      (void)args.insert(args.end(), Xs_.begin(), Xs_.end());
    }

    if (ys_size > 0) {
      (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());
    }

    return node->func_graph()->NewCNode(args);
  }

  void Visit(const CNodePtr &cnode) override {
    // {G, Xs}
    if (cnode->size() < 1 || !IsValueNode<FuncGraph>(cnode->input(0))) {
      return;
    }

    auto &inputs = cnode->inputs();
    fg_ = GetValueNode<FuncGraphPtr>(inputs[0]);
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(Xs_));
  }

  void Reset() {
    Xs_.clear();
    fg_ = nullptr;
  }

 private:
  FuncGraphPtr fg_;
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
    auto new_g1 = call_output_transform_(g1_, ys_size);
    auto new_g2 = call_output_transform_(g2_, ys_size);
    auto sw_node = fg->NewCNode({NewValueNode(prim::kPrimSwitch), x_, NewValueNode(new_g1), NewValueNode(new_g2)});

    std::vector<AnfNodePtr> args{sw_node};
    if (xs_size > 0) {
      (void)args.insert(args.end(), inputs_x.begin() + 1, inputs_x.end());
    }
    if (ys_size > 0) {
      (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());
    }

    return fg->NewCNode(args);
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
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INCORPORATE_CALL_H_
