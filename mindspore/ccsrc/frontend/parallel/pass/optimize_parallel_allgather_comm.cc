/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/optimize_parallel_allgather_comm.h"
#include <memory>
#include <vector>
#include <string>
#include <list>
#include <unordered_map>
#include <algorithm>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"

namespace mindspore {
namespace parallel {
namespace {

bool IsDTypeBitsDecrease(TypeId a, TypeId b) {
  return a == kNumberTypeFloat32 && (b == kNumberTypeFloat16 || b == kNumberTypeBFloat16);
}

void MoveCastBehindAllGather(const FuncGraphPtr &func_graph, const CNodePtr &all_gather_cnode,
                             const CNodePtr &cast_cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(all_gather_cnode);
  MS_EXCEPTION_IF_NULL(cast_cnode);
  auto all_gather_dtype = common::AnfAlgo::GetOutputInferDataType(all_gather_cnode, kIndex0);
  auto cast_dtype = common::AnfAlgo::GetOutputInferDataType(cast_cnode, kIndex0);
  if (!IsDTypeBitsDecrease(all_gather_dtype, cast_dtype)) {
    return;
  }

  // Get operator list from all_gather to cast
  AnfNodePtrList op_list;
  auto cur_node = cast_cnode->input(kIndex1);
  while (cur_node != all_gather_cnode) {
    op_list.push_back(cur_node);
    auto cur_cnode = cur_node->cast<CNodePtr>();
    if (cur_cnode == nullptr) {
      break;
    }
    cur_node = cur_cnode->input(kIndex1);
  }
  if (cur_node != all_gather_cnode) {
    MS_LOG(DEBUG) << "Get op list from all_gather to cast failed.";
    return;
  }
  op_list.push_back(cur_node);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto cast_input_node = cast_cnode->input(kIndex1);
  auto cast_node_users = manager->node_users()[cast_cnode];

  for (const auto &cast_next_node_user_pair : cast_node_users) {
    auto next_cnode = cast_next_node_user_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(next_cnode);
    auto next_index = cast_next_node_user_pair.second;
    manager->SetEdge(next_cnode, next_index, cast_input_node);
  }

  auto all_gather_input_node = all_gather_cnode->input(kIndex1);
  manager->SetEdge(cast_cnode, kIndex1, all_gather_input_node);
  manager->SetEdge(all_gather_cnode, kIndex1, cast_cnode);

  // Update abstract from cast to all_gather
  auto new_cast_abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(cast_dtype),
                                                                 cast_cnode->input(kIndex1)->abstract()->GetShape());
  cast_cnode->set_abstract(new_cast_abs);
  for (auto node : op_list) {
    auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(cast_dtype), node->abstract()->GetShape());
    node->set_abstract(abs);
  }
  return;
}
}  // namespace

void OptimizeParallelAllGatherComm(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (!IsPrimitiveCNode(node, prim::kPrimAllGather) || !common::AnfAlgo::IsFromParallelOptimizer(node)) {
        continue;
      }
      auto all_gather_cnode = node->cast<CNodePtr>();
      auto all_gather_node_user_list = GetOutputNodesWithFilter(all_gather_cnode, [](const AnfNodePtr &node) {
        return IsOneOfPrimitiveCNode(node, {prim::kPrimLoad, prim::kPrimDepend});
      });
      for (auto next_node_pair : all_gather_node_user_list) {
        if (IsPrimitiveCNode(next_node_pair.first, prim::kPrimCast)) {
          MoveCastBehindAllGather(each_graph, all_gather_cnode, next_node_pair.first->cast<CNodePtr>());
        }
      }
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
