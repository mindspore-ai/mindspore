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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ITEM_TUPLE_ELIMINATE_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ITEM_TUPLE_ELIMINATE_H_

#include <vector>
#include <algorithm>

#include "optimizer/irpass.h"
#include "optimizer/optimizer.h"
#include "ir/visitor.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// (a, b, c, ...)[0] => a
// (a, b, c, ...)[1] => b
// {prim::kPrimTupleGetItem, {prim::kPrimMakeTuple, Xs}, C}
class GetitemEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);

    if (is_match_) {
      return tuple_->input(id_);
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      tuple_ = cnode;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (tuple_ != nullptr && IsValueNode<Int32Imm>(vnode)) {
      id_ = IntToSize(GetValue<int>(vnode->value()) + 1);
      if (tuple_->size() > id_) {
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    tuple_ = nullptr;
    is_match_ = false;
  }

 private:
  bool is_match_{false};
  size_t id_{0};
  CNodePtr tuple_{nullptr};
};

// setitem((a, b, c, ...), 0, z) => (z, b, c, ...)
// setitem((a, b, c, ...), 1, z) => (a, z, c, ...)
// {prim::kPrimTupleSetItem, {prim::kPrimMakeTuple, Xs}, C, Z}
class SetitemEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleSetItem, {IsCNode, IsVNode, IsNode})(node);

    auto fg = node->func_graph();
    if (fg != nullptr && z_ != nullptr) {
      args_[id_] = z_;
      return fg->NewCNode(args_);
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_match_) {
      z_ = node;
      return;
    }

    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
      auto &inputs = cnode->inputs();
      (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(args_));
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (args_.size() > 0 && IsValueNode<Int32Imm>(vnode)) {
      id_ = IntToSize(GetValue<int>(vnode->value()) + 1);
      if (id_ < args_.size()) {
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    z_ = nullptr;
    is_match_ = false;
    args_.clear();
  }

 private:
  bool is_match_{false};
  size_t id_{0};
  AnfNodePtr z_{nullptr};
  std::vector<AnfNodePtr> args_{};
};

// {prim::kPrimTupleGetItem, {prim::kPrimTupleSetItem, Y, C1, X}, C2}
class GetSetitemEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);

    auto fg = node->func_graph();
    if (fg != nullptr && key1_ >= 0 && key2_ >= 0) {
      if (key1_ == key2_) {
        return last_;
      }
      return fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), tuple_, c2_});
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimTupleSetItem)) {
      if (cnode->size() < 4) {
        return;
      }

      tuple_ = cnode->input(1);
      last_ = cnode->input(3);

      // key of setitem
      is_in_set_ = true;
      AnfVisitor::Visit(cnode->input(2));
      is_in_set_ = false;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<Int32Imm>(vnode)) {
      auto key = GetValue<int>(vnode->value());
      if (is_in_set_) {
        key1_ = key;
      } else {
        c2_ = vnode;
        key2_ = key;
      }
    }
  }

  void Reset() {
    key1_ = -1;
    key2_ = -1;
    c2_ = nullptr;
    last_ = nullptr;
    tuple_ = nullptr;
    is_in_set_ = false;
  }

 private:
  bool is_in_set_{false};
  int key1_{-1}, key2_{-1};
  AnfNodePtr tuple_{nullptr}, last_{nullptr}, c2_{nullptr};
};

// {prim::kPrimTupleGetItem, {prim::kPrimDepend, X, Y}, C} ->
// {prim::kPrimDepend, {prim::kPrimTupleGetItem, X, C}, Y}
class GetitemDependReorder : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int32Imm>})(node);
    if (x_ == nullptr) {
      return nullptr;
    }

    auto fg = node->func_graph();
    auto item_node = NewCNode({NewValueNode(prim::kPrimTupleGetItem), x_, c_}, fg);
    return NewCNode({NewValueNode(prim::kPrimDepend), item_node, y_}, fg);
  }

  void Visit(const CNodePtr &cnode) override {
    // {prim::kPrimDepend, X, Y}
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && cnode->size() == 3) {
      x_ = cnode->input(1);
      y_ = cnode->input(2);
    }
  }

  void Visit(const ValueNodePtr &vnode) override { c_ = vnode; }

  void Reset() {
    x_ = nullptr;
    y_ = nullptr;
    c_ = nullptr;
  }

 private:
  AnfNodePtr x_{nullptr}, y_{nullptr}, c_{nullptr};
};

class ItemTupleEliminater {
 public:
  ItemTupleEliminater()
      : get_item_eliminater_(), set_item_eliminater_(), get_set_item_eliminater_(), get_item_depend_reorder_() {
    eliminaters_.emplace_back(get_item_eliminater_);
    eliminaters_.emplace_back(set_item_eliminater_);
    eliminaters_.emplace_back(get_set_item_eliminater_);
    eliminaters_.emplace_back(get_item_depend_reorder_);
  }
  ~ItemTupleEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
    AnfNodePtr new_node;
    for (auto &eliminater : eliminaters_) {
      new_node = eliminater(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  GetitemEliminater get_item_eliminater_;
  SetitemEliminater set_item_eliminater_;
  GetSetitemEliminater get_set_item_eliminater_;
  GetitemDependReorder get_item_depend_reorder_;
  std::vector<TransformFuncType> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ITEM_TUPLE_ELIMINATE_H_
