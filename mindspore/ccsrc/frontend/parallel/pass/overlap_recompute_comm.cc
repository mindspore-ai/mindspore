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

#include "frontend/parallel/pass/overlap_recompute_comm.h"
#include <memory>
#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <queue>
#include <utility>
#include "primitive/math_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/sequence_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/utils/utils.h"
#include "include/utils/anfalgo.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace parallel {
namespace {
bool IsNeededCNode(const CNodePtr &cnode) {
  if (!common::AnfAlgo::IsCommunicationOp(cnode)) {
    return false;
  }
  if (!common::AnfAlgo::IsNeededShape(cnode)) {
    return false;
  }
  auto recompute_comm_overlap = MsContext::GetInstance()->get_param<std::string>(MS_CTX_RECOMPUTE_COMM_OVERLAP);
  return common::AnfAlgo::IsNeededOverlapComm(cnode, recompute_comm_overlap);
}

void DoOverlapBetweenTwoList(const FuncGraphPtr &backward_graph, const std::vector<CNodePtr> &comm_node_list1,
                             const std::vector<CNodePtr> &comm_node_list2) {
  size_t min_size = std::min(comm_node_list1.size(), comm_node_list2.size());
  auto manager = backward_graph->manager();
  auto node_users = manager->node_users();
  for (size_t i = 0; i < min_size; ++i) {
    auto comm2_outputs = node_users[comm_node_list2[i]];
    for (const auto &comm2_output : comm2_outputs) {
      common::AnfAlgo::InsertDepend(comm_node_list1[i], comm2_output.first, manager, backward_graph,
                                    "recompute_depend1");
    }
    if (i + 1 >= comm_node_list1.size()) {
      continue;
    }
    auto comm1_outputs = node_users[comm_node_list1[i + 1]];
    for (const auto &comm1_output : comm1_outputs) {
      common::AnfAlgo::InsertDepend(comm_node_list2[i], comm1_output.first, manager, backward_graph,
                                    "recompute_depend2");
    }
  }
}

void OverlapRecomputeCommNodes(const FuncGraphPtr &backward_graph) {
  auto backward_order_cnodes = backward_graph->GetOrderedCnodes();
  std::map<int64_t, std::vector<CNodePtr>> overlap_comm_nodes_map;
  std::map<int64_t, CNodePtr> first_node_map;
  std::map<int64_t, CNodePtr> last_node_map;
  CNodePtrList backward_order_cnode_list(backward_order_cnodes.cbegin(), backward_order_cnodes.cend());
  for (const auto &backward_cnode : backward_order_cnode_list) {
    if (!backward_cnode->HasAttr(kRecomputeSubgraphIdAttr)) {
      continue;
    }
    auto recompute_subgraph_id = GetValue<int64_t>(backward_cnode->GetAttr(kRecomputeSubgraphIdAttr));
    if (IsPrimitiveCNode(backward_cnode) && AnfUtils::IsRealKernel(backward_cnode) &&
        !common::AnfAlgo::IsNopNode(backward_cnode) && !common::AnfAlgo::IsCommunicationOp(backward_cnode) &&
        backward_cnode->size() > kSizeOne) {
      first_node_map.insert({recompute_subgraph_id, backward_cnode});
      last_node_map[recompute_subgraph_id] = backward_cnode;
    }

    if (!IsNeededCNode(backward_cnode)) {
      continue;
    }
    overlap_comm_nodes_map[recompute_subgraph_id].push_back(backward_cnode);
  }
  for (const auto &nodes_pair : first_node_map) {
    auto comm_nodes_v = overlap_comm_nodes_map[nodes_pair.first];
    comm_nodes_v.insert(comm_nodes_v.begin(), nodes_pair.second);
  }

  for (const auto &nodes_pair : last_node_map) {
    auto comm_nodes_v = overlap_comm_nodes_map[nodes_pair.first];
    comm_nodes_v.push_back(nodes_pair.second);
  }

  std::vector<std::vector<CNodePtr>> comm_nodes;
  std::transform(overlap_comm_nodes_map.begin(), overlap_comm_nodes_map.end(), std::back_inserter(comm_nodes),
                 [](const auto &p) { return p.second; });

  for (size_t i = 0; i + 1 < comm_nodes.size(); i += kSizeTwo) {
    DoOverlapBetweenTwoList(backward_graph, comm_nodes[i], comm_nodes[i + 1]);
  }
}
}  // namespace

void OverlapRecomputeComm(const FuncGraphPtr &graph) {
  if ((parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
       parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel)) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto recompute_comm_overlap = ms_context->get_param<std::string>(MS_CTX_RECOMPUTE_COMM_OVERLAP);
  if (recompute_comm_overlap.empty()) {
    return;
  }
  if (ms_context->CellReuseLevel() == CellReuseLevel::kNoCellReuse) {
    return;
  }
  std::vector<FuncGraphPtr> forward_graphs;
  std::vector<FuncGraphPtr> backward_graphs;
  ExtractForwardBackwardGraph(graph, &forward_graphs, &backward_graphs);
  for (const auto &backward_graph : backward_graphs) {
    OverlapRecomputeCommNodes(backward_graph);
  }
}
}  // namespace parallel
}  // namespace mindspore
