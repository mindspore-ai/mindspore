/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_DICT_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_DICT_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <map>

#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
// (a, b, c, ...)['a'] => a
// {prim::kPrimDictGetItem, {prim::kPrimMakeDict, Xs}, C}
class DictGetitemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimDictGetItem, {IsCNode, IsVNode})(node);

    if (id_ != 0) {
      auto value_tuple_cnode = dyn_cast_ptr<CNode>(values_tuple_);
      MS_EXCEPTION_IF_NULL(value_tuple_cnode);
      if (value_tuple_cnode->inputs().size() <= id_) {
        MS_LOG(EXCEPTION) << "The id found is out of value tuple index.";
      }
      return value_tuple_cnode->input(id_);
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeDict)) {
      keys_tuple_ = cnode->input(kIndex1);
      values_tuple_ = cnode->input(kIndex2);
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (keys_tuple_ == nullptr) {
      return;
    }
    auto key_tuple_value = GetValueNode<ValueTuplePtr>(keys_tuple_);
    if (key_tuple_value != nullptr) {
      for (size_t i = 0; i < key_tuple_value->value().size(); ++i) {
        if (*(key_tuple_value->value()[i]) == *(vnode->value())) {
          id_ = i + 1;
          break;
        }
      }
    }
    auto key_make_tuple = keys_tuple_->cast<CNodePtr>();
    if (key_make_tuple == nullptr || !IsPrimitiveCNode(key_make_tuple, prim::kPrimMakeTuple)) {
      return;
    }
    for (size_t i = 1; i < key_make_tuple->inputs().size(); ++i) {
      auto input_i_value = dyn_cast_ptr<ValueNode>(key_make_tuple->input(i));
      MS_EXCEPTION_IF_NULL(input_i_value);
      if (*(input_i_value->value()) == *(vnode->value())) {
        id_ = i;
        break;
      }
    }
  }

  void Reset() {
    id_ = 0;
    keys_tuple_ = nullptr;
    values_tuple_ = nullptr;
  }

 private:
  size_t id_{0};
  AnfNodePtr keys_tuple_{nullptr};
  AnfNodePtr values_tuple_{nullptr};
};

// (a, b, c, ...)['a'] => a
// {prim::kPrimDictGetItem, C1, C}
class DictGetitemConstEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimDictGetItem, {IsVNode, IsVNode})(node);

    if (real_value_ != nullptr) {
      auto out = NewValueNode(real_value_);
      out->set_abstract(real_value_->ToAbstract());
      return out;
    }
    return nullptr;
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<ValueDictionary>(vnode)) {
      dict_ = GetValueNode<ValueDictionaryPtr>(vnode);
      return;
    }
    if (dict_ == nullptr) {
      return;
    }
    for (const auto &key_and_values : dict_->value()) {
      if (*(key_and_values.first) == *(vnode->value())) {
        real_value_ = key_and_values.second;
        break;
      }
    }
  }

  void Reset() {
    real_value_ = nullptr;
    dict_ = nullptr;
  }

 private:
  ValuePtr real_value_;
  ValueDictionaryPtr dict_{nullptr};
};

// (a, b, c, ...)['a'] = z => (z, b, c, ...)
// {prim::kPrimDictSetItem, {prim::kPrimMakeDict, a, b, c, ...}, 'a', z} => {prim::kPrimMakeDict, z, b, c, ...}
// {prim::kPrimDictSetItem, (a, b, c, ...), 'a', z} => {prim::kPrimMakeDict, z, b, c, ...}
class DictSetitemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimDictSetItem, {IsCNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimDictSetItem, {IsVNode, IsVNode, IsNode})(node);

    if (!is_match_) {
      return nullptr;
    }
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    std::vector<AnfNodePtr> key_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
    std::vector<AnfNodePtr> value_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
    GetNewMakeDictInputs(&key_tuple_inputs, &value_tuple_inputs);
    auto key_tuple_node = fg->NewCNodeInOrder(key_tuple_inputs);
    auto value_tuple_node = fg->NewCNodeInOrder(value_tuple_inputs);
    return fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), key_tuple_node, value_tuple_node});
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_match_) {
      z_ = node;
      return;
    }

    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    CNodePtr real_node = cnode;
    while (IsPrimitiveCNode(real_node, prim::kPrimDepend)) {
      auto depend = real_node->cast<CNodePtr>();
      real_node = depend->input(1)->cast<CNodePtr>();
    }
    if (IsPrimitiveCNode(real_node, prim::kPrimMakeDict)) {
      keys_tuple_ = real_node->input(kIndex1);
      values_tuple_ = real_node->input(kIndex2);
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (keys_tuple_ == nullptr && IsValueNode<ValueDictionary>(vnode)) {
      dict_value_ = GetValueNode<ValueDictionaryPtr>(vnode);
      return;
    }
    if (keys_tuple_ == nullptr && dict_value_ == nullptr) {
      return;
    }
    key_ = vnode;
    is_match_ = true;
  }

  void Reset() {
    is_match_ = false;
    keys_tuple_ = nullptr;
    values_tuple_ = nullptr;
    dict_value_ = nullptr;
    key_ = nullptr;
    z_ = nullptr;
  }

 private:
  bool is_match_{false};
  AnfNodePtr keys_tuple_{nullptr};
  AnfNodePtr values_tuple_{nullptr};
  ValueDictionaryPtr dict_value_{nullptr};
  ValueNodePtr key_{nullptr};
  AnfNodePtr z_{nullptr};

  void GetNewMakeDictInputs(std::vector<AnfNodePtr> *key_tuple_inputs, std::vector<AnfNodePtr> *value_tuple_inputs) {
    bool found = false;
    if (keys_tuple_ != nullptr) {
      auto value_tuple_cnode = dyn_cast_ptr<CNode>(values_tuple_);
      MS_EXCEPTION_IF_NULL(value_tuple_cnode);
      auto keys_tuple_value = GetValueNode<ValueTuplePtr>(keys_tuple_);
      if (keys_tuple_value != nullptr) {
        // make_dict(keys_value_tuple, make_tuple(values))
        for (size_t i = 0; i < keys_tuple_value->value().size(); ++i) {
          (void)key_tuple_inputs->emplace_back(NewValueNode(keys_tuple_value->value()[i]));
          if (*(keys_tuple_value->value()[i]) == *(key_->value())) {
            (void)value_tuple_inputs->emplace_back(z_);
            found = true;
          } else {
            (void)value_tuple_inputs->emplace_back(value_tuple_cnode->input(i + 1));
          }
        }
      } else {
        // make_dict(make_tuple(keys), make_tuple(values))
        auto keys_make_tuple = dyn_cast<CNode>(keys_tuple_);
        if (keys_make_tuple == nullptr || !IsPrimitiveCNode(keys_make_tuple, prim::kPrimMakeTuple)) {
          return;
        }
        for (size_t i = 1; i < keys_make_tuple->inputs().size(); ++i) {
          (void)key_tuple_inputs->emplace_back(keys_make_tuple->input(i));
          auto key_input_i_vnode = dyn_cast_ptr<ValueNode>(keys_make_tuple->input(i));
          if (*(key_input_i_vnode->value()) == *(key_->value())) {
            (void)value_tuple_inputs->emplace_back(z_);
            found = true;
          } else {
            (void)value_tuple_inputs->emplace_back(value_tuple_cnode->input(i));
          }
        }
      }
    } else {
      MS_EXCEPTION_IF_NULL(dict_value_);
      for (const auto &key_and_value : dict_value_->value()) {
        (void)key_tuple_inputs->emplace_back(NewValueNode(key_and_value.first));
        if (*(key_and_value.first) == *(key_->value())) {
          (void)value_tuple_inputs->emplace_back(z_);
          found = true;
        } else {
          (void)value_tuple_inputs->emplace_back(NewValueNode(key_and_value.second));
        }
      }
    }
    if (!found) {
      (void)key_tuple_inputs->emplace_back(key_);
      (void)value_tuple_inputs->emplace_back(z_);
    }
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_DICT_ELIMINATE_H_
