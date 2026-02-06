/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MAKE_TUPLE_FROM_FPROP_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MAKE_TUPLE_FROM_FPROP_ELIMINATE_H_

#include <vector>
#include <algorithm>
#include <memory>

#include "frontend/optimizer/irpass.h"
#include "primitive/array_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "include/utils/anfalgo.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {MakeTuple{MakeTuple{loss0, loss1}, Partial{fg, args}}} -> {MakeTuple{loss0, loss1, Partial{fg, args}}}
class MakeTupleFromFpropEliminate : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &opt, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      return nullptr;
    }
    const auto &func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    if (!func_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      return nullptr;
    }
    auto cnode = dyn_cast<CNode>(node);
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    // {prim::kPrimMakeTuple, MakeTupleCNode, PartialCNode}
    constexpr auto expected_input_size = 3;
    if (inputs.size() < expected_input_size) {
      return nullptr;
    }
    const auto &sub_tuple = inputs[1];
    if (!IsPrimitiveCNode(sub_tuple, prim::kPrimMakeTuple)) {
      return nullptr;
    }
    for (size_t i = 2; i < inputs.size(); ++i) {
      if (!IsPrimitiveCNode(inputs[i], prim::kPrimPartial)) {
        return nullptr;
      }
    }
    std::vector<AnfNodePtr> new_tuple_element{NewValueNode(prim::kPrimMakeTuple)};
    const auto &sub_tuple_cnode = dyn_cast<CNode>(sub_tuple);
    MS_EXCEPTION_IF_NULL(sub_tuple_cnode);
    const auto &sub_tuple_elements = sub_tuple_cnode->inputs();
    for (size_t i = 1; i < sub_tuple_elements.size(); i++) {
      (void)new_tuple_element.emplace_back(sub_tuple_elements[i]);
    }
    for (size_t i = 2; i < inputs.size(); ++i) {
      (void)new_tuple_element.emplace_back(inputs[i]);
    }
    const auto &new_node = func_graph->NewCNode(new_tuple_element);
    const auto &manager = opt->resource()->manager();
    ModifyAllUser(node, manager, sub_tuple_elements.size() - 1);
    return new_node;
  }

  // Find out all user of target funcgraph.
  void ModifyAllUser(const AnfNodePtr &node, const FuncGraphManagerPtr &manager, const size_t tuple_size) {
    const auto &fg = node->func_graph();
    const auto &fg_callers = fg->func_graph_cnodes_index();
    if (fg_callers.empty()) {
      return;
    }
    for (auto &caller : fg_callers) {
      const auto &use_node = caller.first->first->cast<CNodePtr>();
      SearchAndReplaceUser(use_node, manager, tuple_size, caller.first->second);
    }
    return;
  }

  // Find out nodes that are using return value of call nodes.
  void SearchAndReplaceUser(const AnfNodePtr &use_node, const FuncGraphManagerPtr &manager, const size_t tuple_size,
                            int index) {
    auto users_sub = manager->node_users()[use_node];
    if (IsPrimitiveCNode(use_node, prim::kPrimSwitch) || IsPrimitiveCNode(use_node, prim::kPrimPartial)) {
      for (auto &user_sub : users_sub) {
        SearchAndReplaceUser(user_sub.first, manager, tuple_size, user_sub.second);
      }
    } else if (index == 0) {
      for (auto &user_sub : users_sub) {
        ReplaceUser(user_sub.first, manager, tuple_size);
      }
    }
  }

  // Modify user
  // Node changed MakeTuple((Tensor1, Tensor2, Tensor3), partial) -> MakeTuple(Tensor1, Tensor2, Tensor3, partial)
  // Getitem(node, 0) -> MakeTuple(Getitem(node, 0), Getitem(node, 1), Getitem(node, 2))
  // Getitem(node, 1) -> Getitem(node, 3)
  void ReplaceUser(const AnfNodePtr &use_node, const FuncGraphManagerPtr &manager, const size_t tuple_size) {
    if (!IsPrimitiveCNode(use_node, prim::kPrimTupleGetItem) ||
        std::find(visit_.begin(), visit_.end(), use_node) != visit_.end()) {
      return;
    }
    const auto &fg = use_node->func_graph();
    const auto &use_cnode = dyn_cast<CNode>(use_node);
    const auto &source_node = use_cnode->input(1);
    const auto index = common::AnfAlgo::GetTupleGetItemOutIndex(use_cnode);
    if (index == 0) {
      std::vector<AnfNodePtr> new_tuple_element{NewValueNode(prim::kPrimMakeTuple)};
      for (size_t i = 0; i < tuple_size; i++) {
        const auto new_element_node =
          fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), source_node, NewValueNode(MakeValue(SizeToLong(i)))});
        (void)visit_.emplace_back(new_element_node);
        (void)new_tuple_element.emplace_back(new_element_node);
      }
      const auto &new_node = fg->NewCNode(new_tuple_element);
      manager->Replace(use_node, new_node);
    } else {
      const auto &new_node = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), source_node,
                                           NewValueNode(MakeValue(SizeToLong(tuple_size + index - 1)))});
      (void)visit_.emplace_back(new_node);
      manager->Replace(use_node, new_node);
    }
  }

 private:
  std::vector<AnfNodePtr> visit_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MAKE_TUPLE_FROM_FPROP_ELIMINATE_H_
