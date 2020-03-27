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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H__

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
class GetitemTransform {
 public:
  GetitemTransform() : cache_() {}
  ~GetitemTransform() = default;

  FuncGraphPtr operator()(const FuncGraphPtr &fg, int idx) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    if (cache.find(idx) == cache.end()) {
      std::ostringstream ss("tp", std::ostringstream::app);
      ss << idx;

      auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      auto output = new_fg->output();
      if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
        auto cnode = output->cast<CNodePtr>();
        auto ids = IntToSize(idx + 1);
        // Inputs should be [make_tuple, item1, item2, ...], so have to offset idx in tuple_getitem by 1.
        if (ids >= cnode->size()) {
          MS_LOG(EXCEPTION) << "index " << ids << " is out of inputs length " << cnode->size();
        }
        new_fg->set_output(cnode->input(ids));
      } else {
        new_fg->set_output(new_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, NewValueNode(idx)}));
      }

      cache[idx] = new_fg;
    }
    return cache[idx];
  }

 private:
  std::unordered_map<FuncGraphPtr, std::unordered_map<int, FuncGraphPtr>> cache_;
};
}  // namespace internal

// {prim::kPrimTupleGetItem, {G, Xs}, C}
class IncorporateGetitem : public AnfVisitor {
 public:
  IncorporateGetitem() : getitem_transform_() {}
  ~IncorporateGetitem() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int32Imm>})(node);

    if (node->func_graph() != nullptr && idx_ >= 0 && fg_ != nullptr) {
      auto new_fg = getitem_transform_(fg_, idx_);
      (void)args_.insert(args_.begin(), NewValueNode(new_fg));
      return node->func_graph()->NewCNode(args_);
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (cnode->size() == 0 || !IsValueNode<FuncGraph>(cnode->input(0))) {
      return;
    }

    auto &inputs = cnode->inputs();
    fg_ = GetValueNode<FuncGraphPtr>(inputs[0]);
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
  }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int>(vnode->value()); }

  void Reset() {
    idx_ = -1;
    fg_ = nullptr;
    args_.clear();
  }

 private:
  int idx_{-1};
  FuncGraphPtr fg_{nullptr};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};

// {prim::kPrimTupleGetItem, {{prim::kPrimSwitch, X, G1, G2}, Xs}, C}
class IncorporateGetitemSwitch : public AnfVisitor {
 public:
  IncorporateGetitemSwitch() : getitem_transform_() {}
  ~IncorporateGetitemSwitch() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    is_in_get_ = true;
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int32Imm>})(node);
    is_in_get_ = false;

    auto fg = node->func_graph();
    if (idx_ == -1 || switch_ == nullptr || fg == nullptr) {
      return nullptr;
    }

    is_in_switch_ = true;
    AnfVisitor::Match(prim::kPrimSwitch, {IsNode, IsValueNode<FuncGraph>, IsValueNode<FuncGraph>})(switch_);
    is_in_switch_ = false;

    if (g2_ == nullptr) {
      return nullptr;
    }

    auto new_g1 = getitem_transform_(g1_, idx_);
    auto new_g2 = getitem_transform_(g2_, idx_);
    auto sw_node = fg->NewCNode({NewValueNode(prim::kPrimSwitch), x_, NewValueNode(new_g1), NewValueNode(new_g2)});
    (void)args_.insert(args_.begin(), sw_node);

    return fg->NewCNode(args_);
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_in_switch_ && x_ == nullptr) {
      x_ = node;
      return;
    }
    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (is_in_get_ && cnode->size() != 0) {
      auto &inputs = cnode->inputs();
      switch_ = inputs[0];
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (is_in_get_) {
      idx_ = GetValue<int>(vnode->value());
    }

    if (is_in_switch_) {
      auto g = GetValueNode<FuncGraphPtr>(vnode);
      if (g1_ == nullptr) {
        g1_ = g;
      } else {
        g2_ = g;
      }
    }
  }

  void Reset() {
    x_ = nullptr;
    g1_ = nullptr;
    g2_ = nullptr;
    switch_ = nullptr;
    args_.clear();
    is_in_get_ = false;
    is_in_switch_ = false;
  }

 private:
  int idx_{-1};
  AnfNodePtr switch_{nullptr}, x_{nullptr};
  FuncGraphPtr g1_{nullptr}, g2_{nullptr};
  bool is_in_get_{false}, is_in_switch_{false};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_
