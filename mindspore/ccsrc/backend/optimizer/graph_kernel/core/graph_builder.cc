/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/core/graph_builder.h"

#include <algorithm>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/core_ops.h"
#include "ir/func_graph.h"
#include "utils/utils.h"

namespace mindspore::graphkernel {
AnfNodePtrList GetOutput(const AnfNodePtrList &nodes, const NodeUsersMap &users,
                         const std::unordered_set<AnfNodePtr> &seen) {
  AnfNodePtrList output;
  if (users.size() == 0) {
    return output;
  }
  for (auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto iter = users.find(node);
    if (iter == users.end()) {
      continue;
    }
    auto &node_users = iter->second;
    const bool has_outer_user = std::any_of(std::begin(node_users), std::end(node_users),
                                            [&seen](const std::pair<AnfNodePtr, int64_t> &u) -> bool {
                                              const bool is_outer_user = (seen.find(u.first) == seen.end());
                                              return is_outer_user;
                                            });
    if (has_outer_user) {
      output.emplace_back(node);
    }
  }
  return output;
}

AnfNodePtr RefSubGraphNode(const FuncGraphPtr &fg, const AnfNodePtr &node, AnfNodePtrList *const inputs_ptr,
                           AnfNodePtrToAnfNodePtrMap *eqv_ptr) {
  auto &input_list = *inputs_ptr;
  auto &eqv = *eqv_ptr;
  if (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) {
    eqv[node] = node;
  } else if (eqv.find(node) == eqv.end()) {
    input_list.push_back(node);
    eqv[node] = fg->add_parameter();
    eqv[node]->set_abstract(node->abstract());
    eqv[node]->set_kernel_info(node->kernel_info_ptr());
  }
  return eqv[node];
}

std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> BuildGraphFromNodes(const AnfNodePtrList &node_list) {
  FuncGraphPtr fg = nullptr;
  {
    // limit the lifetime of guard.
    TraceGuard guard(
      std::make_shared<TraceSegmentTransform>(node_list[0]->cast<CNodePtr>()->func_graph()->debug_info()));
    fg = std::make_shared<FuncGraph>();
  }
  AnfNodePtrList input_list;
  AnfNodePtrToAnfNodePtrMap eqv;
  // Merge CNodes into a AnfGraph that represents a linear instruction segment
  for (auto node : node_list) {
    auto &input_nodes = node->cast<CNodePtr>()->inputs();
    auto fn = input_nodes[0];
    std::vector<AnfNodePtr> new_args{fn};
    if (IsPrimitive(fn, prim::kPrimDepend) && input_nodes.size() >= kDependInputSize &&
        eqv.find(input_nodes[kDependAttachNodeIndex]) == eqv.end()) {
      new_args.emplace_back(RefSubGraphNode(fg, input_nodes[kRealInputIndexInDepend], &input_list, &eqv));
      const size_t value_start_index = 2;
      for (size_t i = value_start_index; i < input_nodes.size(); ++i) {
        new_args.emplace_back(NewValueNode(MakeValue(0)));
      }
    } else {
      (void)std::transform(
        std::begin(input_nodes) + 1, std::end(input_nodes), std::back_inserter(new_args),
        [&fg, &input_list, &eqv](const AnfNodePtr &node) { return RefSubGraphNode(fg, node, &input_list, &eqv); });
    }
    TraceGuard tg(std::make_shared<TraceSegmentTransform>(node->debug_info()));
    eqv[node] = fg->NewCNode(new_args);
    eqv[node]->set_abstract(node->abstract());
    eqv[node]->set_kernel_info(node->kernel_info_ptr());
  }
  std::unordered_set<AnfNodePtr> eqv_keys;
  (void)std::transform(std::begin(eqv), std::end(eqv), std::inserter(eqv_keys, eqv_keys.end()),
                       [](const std::pair<AnfNodePtr, AnfNodePtr> &elem) -> AnfNodePtr { return elem.first; });
  auto mgr = node_list[0]->func_graph()->manager();
  auto outputs = GetOutput(node_list, mgr->node_users(), eqv_keys);
  AnfNodePtr fg_output;
  if (outputs.size() > 1) {
    std::vector<AnfNodePtr> output_args;
    output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
    (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_args),
                         [&eqv](const AnfNodePtr &o) -> AnfNodePtr { return eqv[o]; });
    // Set output for AnfGraph
    fg_output = fg->NewCNode(output_args);
  } else {
    fg_output = eqv[outputs[0]];
  }
  fg->set_output(fg_output);
  return std::make_tuple(fg, input_list, outputs);
}
}  // namespace mindspore::graphkernel
