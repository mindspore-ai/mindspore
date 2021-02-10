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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_

#include <vector>
#include <algorithm>
#include <memory>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

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
    mng_ = optimizer->resource()->manager();
    is_outer_ = true;
    AnfVisitor::Match(prim::kPrimAddN, {IsCNode})(node);
    // do not hold this manager
    mng_ = nullptr;
    if (!is_match_ || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto cnode = node->cast<CNodePtr>();
    auto addn = NewValueNode(GetValueNode(cnode->input(0)));

    // {prim::kPrimMakeTuple, Xs, Ys}, {prim::kPrimMakeTuple, Ys, Xs}
    (void)args_.insert(args_.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto fg = node->func_graph();
    auto make_node = fg->NewCNode(args_);

    return fg->NewCNode({addn, make_node});
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

        (void)Ys_.erase(Ys_.begin());
        (void)std::copy(Xs_.begin(), Xs_.end(), std::back_inserter(args_));
        (void)std::copy(Ys_.begin(), Ys_.end(), std::back_inserter(args_));
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
    is_inner_ = false;
    is_outer_ = false;
    is_match_ = false;
  }

 private:
  FuncGraphManagerPtr mng_{nullptr};
  std::vector<AnfNodePtr> Xs_{}, Ys_{}, args_{};
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
// Akg don't support AddN(ValueNode, Tensor, ...), converted to TensorAdd.
// case0: AddN(inputs)(inputs size < 2) -> error
// case1: AddN(inputs)(all inputs is ValueNode) -> error
// case2: AddN(inputs)(inputs size = 2) -> TensorAdd(Tensor, Tensor)
// case3: AddN(ValueNode, Tensor, Tensor, ...)(has one ValueNode input)
//   -> TensorAdd(ValueNode, AddN(Tensor, Tensor, ...))
class AddNEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    MS_EXCEPTION_IF_NULL(fg);
    auto mng = fg->manager();
    MS_EXCEPTION_IF_NULL(mng);
    if (fg->recursive()) {
      return nullptr;
    }

    auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>("fg"));
    mng->AddFuncGraph(new_fg);
    need_update_ = false;
    bool changed;
    do {
      changed = Process(new_fg);
    } while (changed);

    if (!need_update_) {
      return nullptr;
    } else {
      auto new_sx = inputs;
      new_sx[0] = NewValueNode(new_fg);
      return node->func_graph()->NewCNode(new_sx);
    }
  }

  bool Process(const FuncGraphPtr &func_graph) {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto nodes = TopoSort(func_graph->output());
    bool changed = false;

    for (size_t i = 0; i < nodes.size(); ++i) {
      auto node = nodes[i];
      if (!IsPrimitiveCNode(node, prim::kPrimAddN)) {
        continue;
      }

      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto &tuple_input = cnode->input(1);
      MS_EXCEPTION_IF_NULL(tuple_input);
      auto tuple_input_cnode = tuple_input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_input_cnode);
      auto &tuple_inputs = tuple_input_cnode->inputs();
      if (tuple_inputs.size() < 3) {
        // case0: inputs size < 2, error
        MS_EXCEPTION(ArgumentError) << "Inputs size of AddN less than 2. " << cnode->DebugString(2);
      }

      int64_t valuenode_num = std::accumulate(tuple_inputs.begin() + 1, tuple_inputs.end(), 0,
                                              [](int64_t accumulator, const AnfNodePtr &node) {
                                                if (IsValueNode<tensor::Tensor>(node)) {
                                                  return accumulator + 1;
                                                } else {
                                                  return accumulator;
                                                }
                                              });
      if (LongToSize(valuenode_num) == tuple_inputs.size()) {
        // case1: all inputs is ValueNode, error
        MS_EXCEPTION(ArgumentError) << "All inputs of AddN is ValueNode. " << cnode->DebugString(2);
      }

      if (tuple_inputs.size() == 3) {
        // case2: inputs size = 2, -> TensorAdd(Tensor, Tensor)
        MS_LOG(DEBUG) << "Replace AddN with two inputs with TensorAdd. " << cnode->DebugString(2);
        ValuePtr prim_tensoradd = prim::GetPythonOps("Add", "mindspore.ops.operations");
        std::vector<AnfNodePtr> new_xs{func_graph->NewCNode({NewValueNode(prim_tensoradd)}), tuple_inputs[1],
                                       tuple_inputs[2]};
        mng->Replace(node, func_graph->NewCNode(new_xs));
        changed = true;
        continue;
      }

      auto first_valuenode = std::find_if(tuple_inputs.begin() + 1, tuple_inputs.end(),
                                          [](const AnfNodePtr &node) { return IsValueNode<tensor::Tensor>(node); });
      if (first_valuenode == tuple_inputs.end()) {
        // no ValueNode input found.
        continue;
      } else {
        // case3: has one ValueNode input -> TensorAdd(ValueNode, AddN(Tensor, Tensor, ...))
        std::vector<AnfNodePtr> make_tuple_new_xs{
          NewValueNode(prim::kPrimMakeTuple),
        };
        std::for_each(tuple_inputs.begin() + 1, tuple_inputs.end(),
                      [&make_tuple_new_xs, &first_valuenode](const AnfNodePtr &node) {
                        if (node != *first_valuenode) {
                          make_tuple_new_xs.push_back(node);
                        }
                      });
        ValuePtr prim_addn = prim::GetPythonOps("AddN", "mindspore.ops.operations");
        auto new_addn = func_graph->NewCNode(
          {func_graph->NewCNode({NewValueNode(prim_addn)}), func_graph->NewCNode(make_tuple_new_xs)});
        ValuePtr prim_tensoradd = prim::GetPythonOps("Add", "mindspore.ops.operations");
        auto new_add =
          func_graph->NewCNode({func_graph->NewCNode({NewValueNode(prim_tensoradd)}), *first_valuenode, new_addn});
        (void)mng->Replace(node, new_add);
        changed = true;
        continue;
      }
    }

    need_update_ = need_update_ || changed;
    return changed;
  }

 private:
  bool need_update_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_
