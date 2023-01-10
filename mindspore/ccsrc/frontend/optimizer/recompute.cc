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
#include "include/common/utils/utils.h"
#include "include/common/utils/recompute_helper.h"

namespace mindspore {
namespace opt {
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
  for (const auto &candidate_recomputed_node : candidate_recomputed_nodes) {
    if (visited_nodes.find(candidate_recomputed_node) != visited_nodes.end()) {
      continue;
    }
    mindspore::HashSet<CNodePtr> max_recomputed_sub_graph = {candidate_recomputed_node};
    // Get max continuous recomputed sub-graph.
    GetMaxSubGraph(mng, &max_recomputed_sub_graph, true, true);
    visited_nodes.insert(max_recomputed_sub_graph.cbegin(), max_recomputed_sub_graph.cend());
    // Get the origin recomputed nodes which directly output to the grad nodes.
    mindspore::HashSet<CNodePtr> origin_recomputed_nodes;
    mindspore::HashSet<CNodePtr> target_nodes;
    GetOriginRecomputeAndTargetNodes(mng, max_recomputed_sub_graph, &origin_recomputed_nodes, &target_nodes);
    // Also get the inputs of origin recomputed nodes which eventually output to the grad nodes.
    GetMaxSubGraph(mng, &origin_recomputed_nodes, true, false);

    // Get the inputs of the first target node in the topological sequence. The duplicated recomputed nodes should
    // not be executed until these inputs are ready.
    std::vector<AnfNodePtr> first_target_inputs =
      GetFirstTargetInputs(origin_nodes_topological, origin_recomputed_nodes, target_nodes);
    mindspore::HashMap<CNodePtr, CNodePtr> origin_to_recomputed_nodes;
    // Begin duplicate origin recomputed nodes with each target node.
    DuplicateRecomputedNodes(graph, target_nodes, origin_recomputed_nodes, first_target_inputs,
                             &origin_to_recomputed_nodes);
  }
  // Set need cse attr for doing cse after recompute.
  for (const auto &node : orders) {
    if (WithRecomputedScope(node)) {
      node->AddAttr(kAttrNeedCseAfterRecompute, MakeValue(true));
    }
  }
}
}  // namespace opt
}  // namespace mindspore
