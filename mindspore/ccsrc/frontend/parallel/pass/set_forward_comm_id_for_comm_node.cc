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

#include "frontend/parallel/pass/set_forward_comm_id_for_comm_node.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
bool SetForwardCommIdForCommNode(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  auto graph_set = ForwardGraph(root);
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!IsAutoParallelCareGraph(root) || root->has_flag(SET_PRIMAL_ATTR_FOR_COMM_NODE_RUN_ONCE_ONLY) ||
      graph_set.size() < 1) {
    return changes;
  }

  for (const auto &forward_graph : graph_set) {
    MS_EXCEPTION_IF_NULL(forward_graph);
    const auto &all_nodes = forward_graph->GetOrderedCnodes();
    std::for_each(all_nodes.begin(), all_nodes.end(), [](const CNodePtr &cnode) {
      if (IsCommunicateNode(cnode) && !cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
        cnode->AddPrimalAttr(kPrimalAttrForwardCommNodeUniqueId, MakeValue<std::string>(cnode->UniqueId()));
      }
    });
  }

  // only run once
  root->set_flag(SET_PRIMAL_ATTR_FOR_COMM_NODE_RUN_ONCE_ONLY, true);
  return changes;
}
}  // namespace parallel
}  // namespace mindspore
