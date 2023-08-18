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

#include "plugin/device/ascend/optimizer/ge/expander_fallback.h"
#include <vector>
#include "backend/common/expander/fallback/expander_fallback.h"
#include "include/transform/graph_ir/utils.h"
#include "backend/common/graph_kernel/value_graph_binder.h"

namespace mindspore {
namespace opt {
bool ExpanderFallback::Run(const FuncGraphPtr &graph) {
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!transform::ConvertCheck(node)) {
      auto f = [](const CNodePtr &n) { return true; };
      changed = expander::TryExpandCNode(node, f) || changed;
    }
  }
  if (changed) {
    graphkernel::BindValueToGraph().Run(graph);
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
