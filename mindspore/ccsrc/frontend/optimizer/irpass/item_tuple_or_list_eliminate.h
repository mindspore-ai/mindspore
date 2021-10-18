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
// (a, b, c, ...)[-1] = z => (a, b, c, ...)[length - 1] = z
// [a, b, c, ...][-1] = z => [a, b, c, ...][length - 1] = z
// {prim::kPrimTupleGetItem, T, N}
// {prim::kPrimListGetItem, L, N}
// {prim::kPrimTupleSetItem, T, N, Z}
// {prim::kPrimListSetItem, L, N, Z}
class TupleListConvertItemIndexToPositive : public AnfVisitor {
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
class TupleListGetitemEliminator : public AnfVisitor {
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
class TupleListGetitemConstEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsVNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsVNode, IsVNode})(node);

    if (is_match_) {
      auto out = NewValueNode((*tuple_)[id_]);
      out->set_has_new_value(has_new_value_);
      out->set_abstract((*tuple_)[id_]->ToAbstract());
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

// (a, b, c, ...)[0] = z => (z, b, c, ...)
// [a, b, c, ...][0] = z => [z, b, c, ...]
// {prim::kPrimTupleSetItem, {prim::kPrimMakeTuple, a, b, c, ...}, 0, z} => {prim::kPrimMakeTuple, z, b, c, ...}
// {prim::kPrimListSetItem, {prim::kPrimMakeList, a, b, c, ...}, 0, z} => {prim::kPrimMakeList, z, b, c, ...}
// {prim::kPrimTupleSetItem, (a, b, c, ...), 0, z} => {prim::kPrimMakeTuple, z, b, c, ...}
// {prim::kPrimListSetItem, [a, b, c, ...], 0, z} => {prim::kPrimMakeList, z, b, c, ...}
class TupleListSetitemEliminator : public AnfVisitor {
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
      auto make_tuple = fg->NewCNode(args_);
      // This pass runs after renormalize has finished in pynative mode, so output abstract should be set.
      auto ms_context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(ms_context);
      auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
      if (execution_mode == kPynativeMode) {
        AbstractBasePtrList abs_list;
        for (size_t i = 1; i < args_.size(); ++i) {
          auto abs = args_[i]->abstract();
          MS_EXCEPTION_IF_NULL(abs);
          abs_list.emplace_back(abs->Broaden());
        }
        make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
      }
      return make_tuple;
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
class TupleListGetSetitemEliminator : public AnfVisitor {
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
        MS_EXCEPTION_IF_NULL(tuple_->abstract());
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

// {prim::kPrimTupleGetItem, {prim::kPrimDepend, X, Y}, C} => {prim::kPrimDepend, {prim::kPrimTupleGetItem, X, C}, Y}
// {prim::kPrimListGetItem, {prim::kPrimDepend, X, Y}, C} => {prim::kPrimDepend, {prim::kPrimListGetItem, X, C}, Y}
class TupleListGetitemDependReorder : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &opt, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    if (x_ == nullptr) {
      return nullptr;
    }
    auto mgr = opt->manager();
    MS_EXCEPTION_IF_NULL(mgr);
    auto depend = node->cast<CNodePtr>()->input(1);
    auto depend_cnode = depend->cast<CNodePtr>();
    auto fg = node->func_graph();
    // Avoid generating redundant depend nodes.
    if (ExistUpdateStateUser(mgr, depend)) {
      auto inputs = node->cast<CNodePtr>()->inputs();
      inputs[1] = depend_cnode->input(1);
      auto new_node = fg->NewCNode(inputs);
      new_node->set_abstract(node->abstract());
      return new_node;
    }

    auto item_node = NewCNode({NewValueNode(prim::kPrimTupleGetItem), x_, c_}, fg);
    auto depend_node = NewCNode({depend_cnode->input(0), item_node, y_}, fg);
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

  bool ExistUpdateStateUser(const FuncGraphManagerPtr &mgr, const AnfNodePtr &depend) {
    if (!IsPrimitiveCNode(y_, prim::kPrimUpdateState)) {
      return false;
    }
    auto &node_users = mgr->node_users();
    auto iter = node_users.find(depend);
    if (iter == node_users.end()) {
      return false;
    }
    auto &users = iter->second;
    if (users.size() <= 1) {
      return false;
    }
    bool has_updatestate_user = std::any_of(users.begin(), users.end(), [](const auto &user) {
      return IsPrimitiveCNode(user.first, prim::kPrimUpdateState);
    });
    return has_updatestate_user;
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

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_OR_LIST_ELIMINATE_H_
