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

#include "frontend/jit/ps/static_analysis/event_method.h"

#include "ir/graph_utils.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore {
namespace pipeline {
bool EventMethod(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &all_nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &node_abs = cnode->abstract();
    if (node_abs == nullptr || !node_abs->isa<abstract::AbstractEvent>()) {
      continue;
    }
    const auto &event_abs = node_abs->cast<abstract::AbstractEventPtr>();
    MS_EXCEPTION_IF_NULL(event_abs);
    auto event_id = event_abs->event_id();
    cnode->AddAttr(kAttrEventId, MakeValue(static_cast<uint32_t>(event_id)));
  }
  return false;
}
}  // namespace pipeline
}  // namespace mindspore
