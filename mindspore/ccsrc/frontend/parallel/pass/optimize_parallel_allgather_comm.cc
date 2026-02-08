/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/jit/ps/graph_circle_handler.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

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

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto cast_input_node = cast_cnode->input(kIndex1);

  // Get operator list from all_gather to cast
  auto all_gather_node_users = GetOutputNodesWithFilter(
    all_gather_cnode,
    [](const AnfNodePtr &node) {
      return IsOneOfPrimitiveCNode(node, {prim::kPrimMakeTuple, prim::kPrimDepend, prim::kPrimLoad});
    },
    true);

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
  if (GetCNodePrimitive(all_gather_cnode)->HasAttr(RECOMPUTE)) {
    GetCNodePrimitive(cast_cnode)->AddAttr(RECOMPUTE, GetCNodePrimitive(all_gather_cnode)->GetAttr(RECOMPUTE));
  }
  // Update abstract from cast to all_gather
  auto new_cast_abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(cast_dtype),
                                                                 cast_cnode->input(kIndex1)->abstract()->GetShape());
  cast_cnode->set_abstract(new_cast_abs);
  auto new_all_gather_abs =
    std::make_shared<abstract::AbstractTensor>(TypeIdToType(cast_dtype), all_gather_cnode->abstract()->GetShape());
  all_gather_cnode->set_abstract(new_all_gather_abs);
  for (auto user_pair : all_gather_node_users) {
    if (user_pair.first == cast_cnode) {
      continue;
    }
    if (IsOneOfPrimitiveCNode(user_pair.first, {prim::kPrimUpdateState, prim::kPrimDepend}) &&
        user_pair.second == kIndex2) {
      continue;
    } else if (IsPrimitiveCNode(user_pair.first, prim::kPrimMakeTuple)) {
      auto make_tuple_abs = user_pair.first->abstract();
      MS_EXCEPTION_IF_NULL(make_tuple_abs);
      auto make_tuple_abs_tuple = make_tuple_abs->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(make_tuple_abs_tuple);
      auto abs_list = make_tuple_abs_tuple->ElementsBroaden();
      auto index = user_pair.second - 1;
      auto new_abstract =
        std::make_shared<abstract::AbstractTensor>(TypeIdToType(cast_dtype), abs_list.at(index)->GetShape());
      abs_list.at(index) = new_abstract;
      user_pair.first->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
    } else {
      auto abs =
        std::make_shared<abstract::AbstractTensor>(TypeIdToType(cast_dtype), user_pair.first->abstract()->GetShape());
      user_pair.first->set_abstract(abs);
    }
  }
  return;
}
}  // namespace

void OptimizeParallelAllGatherComm(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  circle_handler::SetAttrToDepend(graph);
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (!IsPrimitiveCNode(node, prim::kPrimAllGather) || !common::AnfAlgo::IsFromParallelOptimizer(node)) {
        continue;
      }
      auto all_gather_cnode = node->cast<CNodePtr>();
      auto all_gather_node_user_list = GetOutputNodesWithFilter(all_gather_cnode, [](const AnfNodePtr &node) {
        return IsOneOfPrimitiveCNode(node, {prim::kPrimLoad, prim::kPrimDepend, prim::kPrimMakeTuple});
      });

      CNodePtr cast_cnode = nullptr;
      for (auto node_user_pair : all_gather_node_user_list) {
        auto user_node = node_user_pair.first;
        if (IsPrimitiveCNode(user_node, prim::kPrimUpdateState) && node_user_pair.second == kIndex2) {
          continue;
        }
        if (IsPrimitiveCNode(user_node, prim::kPrimCast) && cast_cnode == nullptr) {
          cast_cnode = user_node->cast<CNodePtr>();
          continue;
        }
        cast_cnode = nullptr;
        break;
      }

      if (cast_cnode != nullptr) {
        MoveCastBehindAllGather(each_graph, all_gather_cnode, cast_cnode);
      }
    }
  }
  circle_handler::DetectAndRevertGraphCircle(graph, manager, "OptimizeParallelAllGatherComm");
}
}  // namespace parallel
}  // namespace mindspore
