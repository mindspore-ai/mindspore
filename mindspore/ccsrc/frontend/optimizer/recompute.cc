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

#include "frontend/optimizer/recompute.h"
#include <list>
#include <vector>
#include "ir/func_graph.h"
#include "include/utils/utils.h"
#include "include/utils/recompute_helper.h"

namespace mindspore {
namespace opt {
void AddRecomputeSubGraphForBpNodes(const std::vector<CNodePtr> &origin_nodes_topological) {
  std::unordered_map<std::string, CNodePtr> candicate_bp_cnodes_hash_map;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recomputed_block_node_in_orders;
  for (const auto &cnode : origin_nodes_topological) {
    if (cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }

    if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      if (!cnode->HasAttr(kAttrRecomputeSubGraph)) {
        continue;
      }
      auto recompute_block_id = GetValue<size_t>(cnode->GetAttr(kAttrRecomputeSubGraph));
      if (recomputed_block_node_in_orders.find(recompute_block_id) == recomputed_block_node_in_orders.end()) {
        recomputed_block_node_in_orders[recompute_block_id] = {cnode};
      } else {
        recomputed_block_node_in_orders[recompute_block_id].push_back(cnode);
      }
    } else {
      candicate_bp_cnodes_hash_map[GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId))] = cnode;
    }
  }

  for (const auto &recomputed_pair : recomputed_block_node_in_orders) {
    auto recomputed_sub_graph_id = recomputed_pair.first;
    auto recomputed_sub_graph = recomputed_pair.second;
    for (const auto &recomputed_cnode : recomputed_sub_graph) {
      if (!recomputed_cnode->HasPrimalAttr(kPrimalAttrUniqueId)) {
        continue;
      }
      auto unique_id = GetValue<std::string>(recomputed_cnode->GetPrimalAttr(kPrimalAttrUniqueId));
      auto iter = candicate_bp_cnodes_hash_map.find(unique_id);
      if (iter != candicate_bp_cnodes_hash_map.end()) {
        iter->second->AddAttr(kAttrRecomputeSubGraph, MakeValue<size_t>(recomputed_sub_graph_id));
      }
    }
  }
}

void InsertRecomputedNodes(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  SetRecomputedAttr(graph, origin_nodes_topological);
  // Get candidate origin recomputed nodes which have no grad inputs and output to at least one grad node directly.
  std::vector<CNodePtr> candidate_recomputed_nodes = FindCandidateRecomputedNodes(mng, origin_nodes_topological);
  mindspore::HashSet<CNodePtr> visited_nodes;
  size_t recompute_sub_graph_id = 0;
  for (const auto &candidate_recomputed_node : candidate_recomputed_nodes) {
    if (visited_nodes.find(candidate_recomputed_node) != visited_nodes.end()) {
      continue;
    }
    mindspore::HashSet<CNodePtr> max_recomputed_sub_graph = {candidate_recomputed_node};
    // Get max continuous recomputed sub-graph.
    GetMaxSubGraph(mng, &max_recomputed_sub_graph, true, true);
    visited_nodes.insert(max_recomputed_sub_graph.cbegin(), max_recomputed_sub_graph.cend());
    for (const auto &sub_graph_node : max_recomputed_sub_graph) {
      sub_graph_node->AddAttr(kAttrRecomputeSubGraph, MakeValue(recompute_sub_graph_id));
    }
    recompute_sub_graph_id++;
    // Get the origin recomputed nodes which directly output to the grad nodes.
    mindspore::HashSet<CNodePtr> origin_recomputed_nodes;
    mindspore::HashSet<CNodePtr> target_nodes;
    GetOriginRecomputeAndTargetNodes(mng, max_recomputed_sub_graph, &origin_recomputed_nodes, &target_nodes);
    // Also get the inputs of origin recomputed nodes which eventually output to the grad nodes.
    GetMaxSubGraph(mng, &origin_recomputed_nodes, true, false);

    // Get the inputs of the first target node in the topological sequence. The duplicated recomputed nodes should
    // not be executed until these inputs are ready.
    std::vector<AnfNodePtr> first_target_inputs =
      GetFirstTargetInputs(origin_nodes_topological, max_recomputed_sub_graph, origin_recomputed_nodes, target_nodes);
    static bool warning_printed = false;
    if (first_target_inputs.empty() && !warning_printed) {
      warning_printed = true;
      MS_LOG(WARNING) << "Can not find the nodes to depend, please check the recompute strategy.";
    }
    mindspore::HashMap<CNodePtr, CNodePtr> origin_to_recomputed_nodes;
    mindspore::HashMap<CNodePtr, CNodePtr> origin_to_new_target_nodes;
    // Begin duplicate origin recomputed nodes with each target node.
    DuplicateRecomputedNodes(graph, target_nodes, origin_recomputed_nodes, first_target_inputs,
                             &origin_to_new_target_nodes, &origin_to_recomputed_nodes);
    // Update the topological nodes.
    for (size_t i = 0; i < origin_nodes_topological.size(); ++i) {
      auto iter = origin_to_new_target_nodes.find(origin_nodes_topological[i]);
      if (iter != origin_to_new_target_nodes.end()) {
        origin_nodes_topological[i] = iter->second;
      }
    }
  }
  // Set need cse attr for doing cse after recompute.
  for (const auto &node : orders) {
    if (WithRecomputedScope(node)) {
      node->AddAttr(kAttrNeedCseAfterRecompute, MakeValue(true));
    }
  }
  std::list<CNodePtr> new_orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> new_origin_nodes_topological(new_orders.cbegin(), new_orders.cend());
  AddRecomputeSubGraphForBpNodes(new_origin_nodes_topological);
}
}  // namespace opt
}  // namespace mindspore
