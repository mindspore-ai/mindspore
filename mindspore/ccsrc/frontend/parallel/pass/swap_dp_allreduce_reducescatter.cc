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

#include "frontend/parallel/pass/swap_dp_allreduce_reducescatter.h"
#include <vector>
#include <list>
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/node_check.h"

namespace mindspore {
namespace parallel {
void SwapDpAllreduceReduceScatter(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  for (const auto &each_graph : manager->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (!IsFromParallelOptimizerRs(node)) {
        continue;
      }
      if (!IsFromGradMirrorAR(node->input(kIndex1))) {
        continue;
      }
      auto allreduce_cnode = node->input(kIndex1)->cast<CNodePtr>();
      MS_LOG(INFO) << "Swap reduce_scatter and all_reduce in dp group";
      auto origin_rs_abs = node->abstract()->Clone();
      auto rs_users = manager->node_users()[node];
      manager->SetEdge(node, kIndex1, allreduce_cnode->input(kIndex1));
      manager->SetEdge(allreduce_cnode, kIndex1, node);
      for (const auto &rs_user : rs_users) {
        manager->SetEdge(rs_user.first, rs_user.second, allreduce_cnode);
      }
      allreduce_cnode->set_abstract(origin_rs_abs);
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
