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
#include "backend/common/graph_kernel/core/graph_kernel_op_combiner.h"

namespace mindspore::graphkernel {
bool GraphKernelOpCombiner::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto cb = Callback::Instance();
  auto nodes = TopoSort(func_graph->get_return());
  auto changed = false;
  for (auto node : nodes) {
    if (node->cast<CNodePtr>() == nullptr || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto node_name = node->fullname_with_scope();
    auto res = ConcatParallelMatMul(node, min_ops_to_combine_, default_layout_to_combine_, func_graph);
    if (res != nullptr) {
      changed = true;
      mng->RemoveRoots();
      mng->KeepRoots({func_graph});
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
