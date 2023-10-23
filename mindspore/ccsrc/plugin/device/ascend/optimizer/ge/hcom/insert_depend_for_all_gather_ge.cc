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

#include "plugin/device/ascend/optimizer/ge/hcom/insert_depend_for_all_gather_ge.h"
#include <unordered_map>
#include <queue>
#include <vector>
#include <utility>
#include "ops/framework_ops.h"
#include "ops/array_ops.h"
#include "ops/other_ops.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr GetInputNodeWithFilter(const AnfNodePtr &node,
                                  std::function<std::pair<bool, size_t>(const CNodePtr &)> filter) {
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    if (!queue_end->isa<CNode>()) {
      return queue_end;
    }
    auto cnode_queue_end = queue_end->cast<CNodePtr>();
    auto filter_res = filter(cnode_queue_end);
    if (!filter_res.first) {
      return queue_end;
    }
    anf_queue.push(cnode_queue_end->input(filter_res.second));
  }
  return node;
}

CNodePtr CreateDepend(const AnfNodePtr &first_input, const AnfNodePtr &second_input, const FuncGraphPtr &graph) {
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), first_input,
                                    second_input};
  auto new_input = graph->NewCNode(inputs);
  new_input->set_abstract(first_input->abstract());
  return new_input;
}

bool InsertDepend(const FuncGraphPtr &graph, const std::vector<CNodePtr> &allgather_with_output_order,
                  const std::unordered_map<CNodePtr, CNodePtr> &allgather_output_another_input) {
  bool changed = false;
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (size_t i = 0; i + 1 < allgather_with_output_order.size(); ++i) {
    auto current_ag_node = allgather_with_output_order[i];
    auto next_ag_node = allgather_with_output_order[i + 1];
    auto next_ag_input = next_ag_node->input(1);
    // next_ag_input -> current_ag_node_output
    auto depend2 = CreateDepend(current_ag_node, next_ag_input, graph);
    manager->Replace(current_ag_node, depend2);
    depend2->AddAttr("opt_shard_depend2", MakeValue(true));
    // current_ag_node -> next_ag_node
    auto depend1 = CreateDepend(next_ag_node->input(1), current_ag_node, graph);
    manager->SetEdge(next_ag_node, 1, depend1);
    depend1->AddAttr("opt_shard_depend1", MakeValue(true));
    changed = true;
    // allgather_output_another_input -> allgather
    if (allgather_output_another_input.count(current_ag_node) == 0) {
      continue;
    }

    auto cur_node_users = manager->node_users()[current_ag_node];
    for (const auto &allgather_node_user : cur_node_users) {
      if (!IsPrimitiveCNode(allgather_node_user.first) ||
          (IsPrimitiveCNode(allgather_node_user.first, prim::kPrimDepend) && allgather_node_user.second == 2)) {
        continue;
      }
      auto allgather_node_user_cnode = allgather_node_user.first->cast<CNodePtr>();
      auto depend3 = CreateDepend(current_ag_node, allgather_output_another_input.at(current_ag_node), graph);
      manager->SetEdge(allgather_node_user_cnode, allgather_node_user.second, depend3);
      depend3->AddAttr("opt_shard_depend3", MakeValue(true));
    }

    // allgather_output_another_input -> next_ag
    auto depend4 = CreateDepend(next_ag_node->input(1), allgather_output_another_input.at(current_ag_node), graph);
    depend4->AddAttr("opt_shard_depend4", MakeValue(true));
    manager->SetEdge(next_ag_node, 1, depend4);
  }
  return changed;
}
}  // namespace

bool InsertDependForAllGatherGe::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<CNodePtr> allgather_with_output_order;
  std::unordered_map<CNodePtr, CNodePtr> allgather_output_another_input;
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    bool is_recompute = cnode->GetAttr(kAttrDuplicated) != nullptr && GetValue<bool>(cnode->GetAttr(kAttrDuplicated));
    if (is_recompute) {
      continue;
    }
    size_t ag_index = 0;
    CNodePtr ag_node = nullptr;
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto pre_cnode = GetInputNodeWithFilter(cnode->input(i), [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                      IsPrimitiveCNode(cnode, prim::kPrimDepend);
        return std::make_pair(filter, 1);
      });
      if (!IsPrimitiveCNode(pre_cnode, prim::kPrimAllGather) ||
          !common::AnfAlgo::IsFromParallelOptimizer(pre_cnode->cast<CNodePtr>())) {
        continue;
      }
      if (std::find(allgather_with_output_order.begin(), allgather_with_output_order.end(),
                    pre_cnode->cast<CNodePtr>()) != allgather_with_output_order.end()) {
        continue;
      }
      allgather_with_output_order.push_back(pre_cnode->cast<CNodePtr>());
      ag_index = i;
      ag_node = pre_cnode->cast<CNodePtr>();
    }
    for (size_t i = 1; i < cnode->size(); ++i) {
      if (ag_index > 0 && i != ag_index && ag_node && IsPrimitiveCNode(cnode->input(i))) {
        allgather_output_another_input[ag_node] = cnode->input(i)->cast<CNodePtr>();
      }
    }
  }
  return InsertDepend(graph, allgather_with_output_order, allgather_output_another_input);
}
}  // namespace opt
}  // namespace mindspore
