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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENVIRON_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENVIRON_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/inline.h"
#include "frontend/optimizer/optimizer.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
constexpr int kEnvironGetSetInputSize = 4;
constexpr int kEnvironOffset = 1;
constexpr int kSymbolicKeyOffset = 2;
constexpr int kValueOffset = 3;

AnfNodePtr GetIndexedEnvironValueNode(const FuncGraphPtr &fg, const AnfNodePtr &origin_value_node,
                                      const std::size_t index) {
  AnfNodePtr new_value_node;
  if (IsValueNode<ValueTuple>(origin_value_node)) {
    auto origin_value_tuple = GetValueNode<ValueTuplePtr>(origin_value_node);
    if (index >= origin_value_tuple->size()) {
      MS_LOG(EXCEPTION) << "Index: " << index << " is greater than Value size: " << origin_value_tuple->size()
                        << ", Default Value: " << origin_value_node->ToString();
    }
    new_value_node = NewValueNode((*origin_value_tuple)[index]);
  } else {
    new_value_node = fg->NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), origin_value_node, NewValueNode(MakeValue(static_cast<int64_t>(index)))});
  }
  return new_value_node;
}
}  // namespace internal

// {prim::kPrimEnvironGet, C1, C2, Y} -> Y
class EnvironGetEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode c1;
    PatternNode c2;
    PatternNode y;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimEnvironGet, c1, c2, y), y,
                     (IsNewEnvironNode(c1.GetNode(node)) && IsVNode(c2.GetNode(node))));
    return nullptr;
  }
};

// {prim::kPrimEnvironGet, {prim::kPrimEnvironAdd, X, Y}, C, Z} ->
// {prim::GetPythonOps("hyper_add"), {prim::kPrimEnvironGet, X, C, Z}, {prim::kPrimEnvironGet, Y, C, Z}}
class EnvironGetAddEliminater : public AnfVisitor {
 public:
  EnvironGetAddEliminater() {
    py::gil_scoped_acquire gil;
    PrimHyperAdd_ = prim::GetPythonOps("hyper_add");
  }
  ~EnvironGetAddEliminater() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    auto IsAddCNode = [](const AnfNodePtr &node) -> bool {
      return IsPrimitiveCNode(node, prim::kPrimEnvironAdd) && node->cast<CNodePtr>()->size() == 3;
    };
    AnfVisitor::Match(prim::kPrimEnvironGet, {IsAddCNode, IsVNode, IsNode})(node);

    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {prim::kPrimEnvironGet, {...}, C, Z}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(inp1);
    auto c = cnode->input(2);
    auto z = cnode->input(3);

    // {prim::kPrimEnvironAdd, X, Y}
    auto x = inp1->input(1);
    auto y = inp1->input(2);

    auto fg = node->func_graph();
    auto xcz = fg->NewCNode({NewValueNode(prim::kPrimEnvironGet), x, c, z});
    auto ycz = fg->NewCNode({NewValueNode(prim::kPrimEnvironGet), y, c, z});

    return fg->NewCNode({NewValueNode(PrimHyperAdd_), xcz, ycz});
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
  ValuePtr PrimHyperAdd_;
};

// {prim::kPrimEnvironGet, {prim::kPrimEnvironSet, X, C1, Y}, C2, Z}
class EnvironGetSetEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    is_match_ = false;
    auto IsSetCNode = [](const AnfNodePtr &node) -> bool {
      if (!IsPrimitiveCNode(node, prim::kPrimEnvironSet)) {
        return false;
      }

      // {prim::kPrimEnvironSet, X, C1, Y}
      auto &inputs = node->cast<CNodePtr>()->inputs();
      if (inputs.size() != 4) {
        return false;
      }

      return IsValueNode<SymbolicKeyInstance>(inputs[2]);
    };
    AnfVisitor::Match(prim::kPrimEnvironGet, {IsSetCNode, IsValueNode<SymbolicKeyInstance>, IsNode})(node);

    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {prim::kPrimEnvironGet, {...}, C2, Z}
    auto cnode = node->cast<CNodePtr>();
    auto inp1 = cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(inp1);
    auto key2 = cnode->input(2);
    auto c2 = GetValueNode<SymbolicKeyInstancePtr>(key2);
    auto default_v = cnode->input(3);

    // {prim::kPrimEnvironSet, X, C1, Y}
    AnfNodePtr environ_node = inp1->input(1);
    auto c1 = GetValueNode<SymbolicKeyInstancePtr>(inp1->input(2));
    auto last_set = inp1->input(3);

    if (*c1 == *c2) {
      return last_set;
    }

    while (IsPrimitiveCNode(environ_node, prim::kPrimEnvironSet)) {
      // {prim::kPrimEnvironSet, environ, symbolickey, value}
      auto &inputs = environ_node->cast<CNodePtr>()->inputs();
      if (inputs.size() != internal::kEnvironGetSetInputSize) {
        MS_LOG(WARNING) << "Input size should be 4";
        return nullptr;
      }
      if (!IsValueNode<SymbolicKeyInstance>(inputs[internal::kSymbolicKeyOffset])) {
        MS_LOG(DEBUG) << "Input 2 is not a SymbolicKeyInstance?";
        return nullptr;
      }

      environ_node = inputs[internal::kEnvironOffset];
      last_set = inputs[internal::kValueOffset];
      auto symbolic_c1 = GetValueNode<SymbolicKeyInstancePtr>(inputs[internal::kSymbolicKeyOffset]);
      if (*symbolic_c1 == *c2) {
        return last_set;
      }
    }

    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimEnvironGet), environ_node, key2, default_v});
  }

  void Visit(const AnfNodePtr &) override { is_match_ = true; }

 private:
  bool is_match_{false};
};

// {prim::kPrimEnvironGet, {prim::kPrimDepend, X1, X2}, item, dflt} ->
// {prim::kPrimDepend, {prim::kPrimEnvironGet, X1, item, dflt}, X2}
class EnvironGetDependSwap : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }
    ScopePtr scope = node->cast<CNodePtr>()->scope();
    ScopeGuard scope_guard(scope);

    PatternNode x1;
    PatternNode x2;
    PatternNode item;
    PatternNode dflt;
    MATCH_REPLACE(node, PPrimitive(prim::kPrimEnvironGet, PPrimitive(prim::kPrimDepend, x1, x2), item, dflt),
                  PPrimitive(prim::kPrimDepend, PPrimitive(prim::kPrimEnvironGet, x1, item, dflt), x2));
    return nullptr;
  }
};

// {prim::kPrimEnvironAdd, C1, X} -> X
// {prim::kPrimEnvironAdd, X, C1} -> X
class EnvironAddConstEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode c1;
    PatternNode x;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimEnvironAdd, c1, x), x, (IsNewEnvironNode(c1.GetNode(node))));
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimEnvironAdd, x, c1), x, (IsNewEnvironNode(c1.GetNode(node))));
    return nullptr;
  }
};

// {prim::kPrimEnvironSet, E, K, V} ->
//     E1 = {prim::kPrimEnvironSet, E,  K1, V1},
//     E2 = {prim::kPrimEnvironSet, E1, K2, V2},
//     ...
// {prim::kPrimEnvironGet, E, K, V} ->
//     v1 = {prim::kPrimEnvironGet, E, K1, default_v1},
//     v2 = {prim::kPrimEnvironGet, E, K2, devault_v2},
//     ...
//     v_tuple = {prim::kPrimMakeTuple, v1, v2, ...}
class SplitEnvironGetSetWithTupleValue : public AnfVisitor {
 public:
  SplitEnvironGetSetWithTupleValue() = default;
  ~SplitEnvironGetSetWithTupleValue() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!(IsPrimitiveCNode(node, prim::kPrimEnvironSet) || IsPrimitiveCNode(node, prim::kPrimEnvironGet)) ||
        node->func_graph() == nullptr) {
      return nullptr;
    }
    // {prim::kPrimEnvironSet, E, key, node_with_abstract_is_tuple} or
    // {prim::kPrimEnvironGet, E, key, node_with_abstract_is_tuple}
    const auto &cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    auto &environ_node = inputs[internal::kEnvironOffset];
    const auto &origin_value_node = inputs[internal::kValueOffset];
    const auto &origin_key_node = GetValueNode<SymbolicKeyInstancePtr>(inputs[internal::kSymbolicKeyOffset]);

    if (origin_key_node == nullptr || origin_value_node->abstract() == nullptr ||
        !origin_value_node->abstract()->isa<abstract::AbstractTuple>()) {
      return nullptr;
    }

    const auto &origin_value_abstract = origin_value_node->abstract()->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(origin_value_abstract);

    AnfNodePtr prev_environ_node = environ_node;
    auto fg = node->func_graph();

    if (IsPrimitiveCNode(node, prim::kPrimEnvironSet)) {
      CNodePtr new_cnode = cnode;
      // Cascade the split CNode of EnvironSet.
      for (std::size_t index = 0; index < origin_value_abstract->elements().size(); ++index) {
        auto new_key = std::make_shared<SymbolicKeyInstance>(
          origin_key_node->node(), origin_value_abstract->elements()[index], static_cast<int64_t>(index));
        AnfNodePtr new_value_node = internal::GetIndexedEnvironValueNode(fg, origin_value_node, index);

        new_cnode = fg->NewCNode({inputs[0], prev_environ_node, NewValueNode(new_key), new_value_node});
        prev_environ_node = new_cnode;
      }
      return new_cnode;
    } else {
      // MakeTuple the split CNode of EnvironGet.
      AnfNodePtrList tuple_item_list{NewValueNode(prim::kPrimMakeTuple)};
      for (std::size_t index = 0; index < origin_value_abstract->elements().size(); ++index) {
        auto new_key = std::make_shared<SymbolicKeyInstance>(
          origin_key_node->node(), origin_value_abstract->elements()[index], static_cast<int64_t>(index));
        AnfNodePtr new_value_node = internal::GetIndexedEnvironValueNode(fg, origin_value_node, index);
        auto new_item_cnode = fg->NewCNode({inputs[0], environ_node, NewValueNode(new_key), new_value_node});
        tuple_item_list.push_back(new_item_cnode);
      }
      auto new_cnode = fg->NewCNode(tuple_item_list);
      return new_cnode;
    }
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ENVIRON_ELIMINATE_H_
