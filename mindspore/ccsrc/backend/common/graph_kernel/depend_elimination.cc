/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/depend_elimination.h"
#include "include/common/utils/utils.h"

namespace mindspore::graphkernel {
bool DependElimination::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  // Depend node will be replaced by its first input valuenode when it has two same inputs.
  for (auto &node : todos) {
    if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();
      // Currently can only handle depend with 2 inputs, e.g. CNode(Depend, %1, %2)
      if (inputs.size() != kDependInputSize) {
        MS_LOG(DEBUG) << "Depend node does not have " << kDependInputSize << " inputs.";
        continue;
      }
      if (inputs[kRealInputIndexInDepend] == inputs[kDependAttachNodeIndex]) {
        (void)mng->Replace(node, inputs[kRealInputIndexInDepend]);
        MS_LOG(INFO) << "Depend node has been replaced by " << inputs[kRealInputIndexInDepend];
      }
    }
  }
  return true;
}
}  // namespace mindspore::graphkernel
