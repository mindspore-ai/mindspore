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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_OR_LIST_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_OR_LIST_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
// (a, b, c, ...)[-1] => (a, b, c, ...)[length-1]
// [a, b, c, ...][-1] => [a, b, c, ...][length-1]
// {prim::kPrimTupleGetItem, T, N}
// {prim::kPrimListGetItem, L, N}
// setitem((a, b, c, ...), -1, z) => setitem((a, b, c, ...), length - 1, z)
// setitem([a, b, c, ...], -1, z) => setitem([a, b, c, ...], length - 1, z)
// {prim::kPrimTupleSetItem, T, N, Z}
// {prim::kPrimListSetItem, L, N, Z}
class ConvertItemIndexToPositive : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimTupleSetItem, {IsCNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimListSetItem, {IsCNode, IsVNode, IsNode})(node);

    FuncGraphPtr fg = node->func_graph();
    if (is_match_ && fg != nullptr) {
      auto inputs = node->cast<CNodePtr>()->inputs();
      inputs[2] = NewValueNode(id_);
      return fg->NewCNode(inputs);
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_match_) {
      return;
    }

    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override { sequeue_ = cnode; }

  void Visit(const ValueNodePtr &vnode) override {
    if (sequeue_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        auto sequeue_abstract = sequeue_->abstract()->cast<abstract::AbstractSequeuePtr>();
        if (sequeue_abstract == nullptr) {
          return;
        }
        id_ = idx + sequeue_abstract->size();
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    sequeue_ = nullptr;
    is_match_ = false;
  }

 private:
  bool is_match_{false};
  int64_t id_{0};
  CNodePtr sequeue_{nullptr};
};

// (a, b, c, ...)[0] => a
// (a, b, c, ...)[1] => b
// {prim::kPrimTupleGetItem, {prim::kPrimMakeTuple, Xs}, C}
// {prim::kPrimListGetItem, {prim::kPrimMakeList, Xs}, C}
class GetitemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsVNode})(node);

    if (is_match_) {
      return tuple_->input(id_);
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
      tuple_ = cnode;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (tuple_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        idx = idx + tuple_->size() - 1;
      }
      id_ = LongToSize(idx + 1);
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

// (a, b, c, ...)[0] => a
// (a, b, c, ...)[1] => b
// {prim::kPrimTupleGetItem, C1, C}
// {prim::kPrimListGetItem, C1, C}
class GetitemConstEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsVNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsVNode, IsVNode})(node);

    if (is_match_) {
      auto out = NewValueNode((*tuple_)[id_]);
      out->set_has_new_value(has_new_value_);
      return out;
    }
    return nullptr;
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<ValueTuple>(vnode)) {
      tuple_ = GetValueNode<ValueTuplePtr>(vnode);
      has_new_value_ = vnode->has_new_value();
    }
    if (tuple_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        idx = idx + tuple_->size();
      }
      id_ = LongToSize(idx);
      if (id_ < tuple_->size()) {
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
  ValueTuplePtr tuple_{nullptr};
  bool has_new_value_{false};
};

// setitem((a, b, c, ...), 0, z) => (z, b, c, ...)
// setitem((a, b, c, ...), 1, z) => (a, z, c, ...)
// {prim::kPrimTupleSetItem, {prim::kPrimMakeTuple, a, b, c, ...}, 0, z} => {prim::kPrimMakeTuple, z, b, c, ...}
// {prim::kPrimListSetItem, {prim::kPrimMakeList, a, b, c, ...}, 0, z} => {prim::kPrimMakeList, z, b, c, ...}
// {prim::kPrimTupleSetItem, (a, b, c, ...), 0, z} => {prim::kPrimMakeTuple, z, b, c, ...}
// {prim::kPrimListSetItem, [a, b, c, ...], 0, z} => {prim::kPrimMakeList, z, b, c, ...}
class SetitemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleSetItem, {IsCNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimListSetItem, {IsCNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimTupleSetItem, {IsVNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimListSetItem, {IsVNode, IsVNode, IsNode})(node);

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
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
      auto &inputs = cnode->inputs();
      (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(args_));
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (args_.empty() && IsValueNode<ValueTuple>(vnode)) {
      auto tuple = GetValueNode<ValueTuplePtr>(vnode);
      if (tuple != nullptr) {
        args_.emplace_back(NewValueNode(prim::kPrimMakeTuple));
        for (auto &val : tuple->value()) {
          auto val_node = std::make_shared<ValueNode>(val);
          val_node->set_abstract(val->ToAbstract());
          args_.emplace_back(val_node);
        }
      }
    } else if (args_.empty() && IsValueNode<ValueList>(vnode)) {
      auto list = GetValueNode<ValueListPtr>(vnode);
      if (list != nullptr) {
        args_.emplace_back(NewValueNode(prim::kPrimMakeList));
        for (auto &val : list->value()) {
          auto val_node = std::make_shared<ValueNode>(val);
          val_node->set_abstract(val->ToAbstract());
          args_.emplace_back(val_node);
        }
      }
    } else if (!args_.empty() && IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        idx = idx + args_.size() - 1;
      }
      id_ = LongToSize(idx + 1);
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
// {prim::kPrimListGetItem, {prim::kPrimListSetItem, Y, C1, X}, C2}
class GetSetitemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsVNode})(node);

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
    if (IsPrimitiveCNode(cnode, prim::kPrimTupleSetItem) || IsPrimitiveCNode(cnode, prim::kPrimListSetItem)) {
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
    if (tuple_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      auto key = GetValue<int64_t>(vnode->value());
      if (key < 0) {
        auto sequeue_abstract = tuple_->abstract()->cast<abstract::AbstractSequeuePtr>();
        if (sequeue_abstract == nullptr) {
          return;
        }
        key = key + sequeue_abstract->size();
      }
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
  int64_t key1_{-1}, key2_{-1};
  AnfNodePtr tuple_{nullptr}, last_{nullptr}, c2_{nullptr};
};

// {prim::kPrimTupleGetItem, {prim::kPrimDepend, X, Y}, C} ->
// {prim::kPrimDepend, {prim::kPrimTupleGetItem, X, C}, Y}
// {prim::kPrimListGetItem, {prim::kPrimDepend, X, Y}, C} ->
// {prim::kPrimDepend, {prim::kPrimListGetItem, X, C}, Y}
class GetitemDependReorder : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    if (x_ == nullptr) {
      return nullptr;
    }

    auto fg = node->func_graph();
    auto item_node = NewCNode({NewValueNode(prim::kPrimTupleGetItem), x_, c_}, fg);
    auto depend_node = NewCNode({NewValueNode(prim::kPrimDepend), item_node, y_}, fg);
    auto abs = x_->abstract();
    if (abs == nullptr) {
      return depend_node;
    }
    auto idx_value = GetValueNode<Int64ImmPtr>(c_);
    MS_EXCEPTION_IF_NULL(idx_value);
    int64_t idx = idx_value->value();
    if (abs->isa<abstract::AbstractTuple>()) {
      auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
      if (idx < 0) {
        idx += abs_tuple->elements().size();
      }
      if (idx < 0 || LongToSize(idx) >= abs_tuple->elements().size()) {
        MS_LOG(EXCEPTION) << "The idx value " << idx << " of tuple_getitem node " << c_->DebugString()
                          << " is out of range.";
      }
      item_node->set_abstract(abs_tuple->elements()[idx]);
    } else {
      item_node->set_abstract(abs);
    }
    depend_node->set_abstract(item_node->abstract());
    return depend_node;
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

class ItemTupleOrListEliminator : public OptimizerCaller {
 public:
  ItemTupleOrListEliminator()
      : get_item_eliminator_(std::make_shared<GetitemEliminator>()),
        get_item_const_eliminator_(std::make_shared<GetitemConstEliminator>()),
        set_item_eliminator_(std::make_shared<SetitemEliminator>()),
        get_set_item_eliminator_(std::make_shared<GetSetitemEliminator>()),
        get_item_depend_reorder_(std::make_shared<GetitemDependReorder>()),
        convert_item_index_to_positive_(std::make_shared<ConvertItemIndexToPositive>()) {
    eliminators_.emplace_back(get_item_eliminator_);
    eliminators_.emplace_back(get_item_const_eliminator_);
    eliminators_.emplace_back(set_item_eliminator_);
    eliminators_.emplace_back(get_set_item_eliminator_);
    eliminators_.emplace_back(get_item_depend_reorder_);
    eliminators_.emplace_back(convert_item_index_to_positive_);
  }
  ~ItemTupleOrListEliminator() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    AnfNodePtr new_node;
    for (auto &eliminator : eliminators_) {
      new_node = (*eliminator)(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  OptimizerCallerPtr get_item_eliminator_, get_item_const_eliminator_, set_item_eliminator_, get_set_item_eliminator_,
    get_item_depend_reorder_, convert_item_index_to_positive_;
  std::vector<OptimizerCallerPtr> eliminators_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_OR_LIST_ELIMINATE_H_
