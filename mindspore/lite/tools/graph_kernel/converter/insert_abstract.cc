/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "tools/graph_kernel/converter/insert_abstract.h"

#include "utils/anf_utils.h"
#include "common/graph_kernel/core/graph_kernel_callback.h"

namespace mindspore::graphkernel {
bool InsertAbstract::Run(const FuncGraphPtr &func_graph) {
  bool changed = false;
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  for (auto &node : todos) {
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      auto cnode = node->cast<CNodePtr>();
      auto input2 = cnode->input(kInputNodeOutputIndexInTupleGetItem);
      auto item_idx = LongToSize(AnfUtils::GetIntValue(input2));
      auto abs_tuple = dyn_cast<abstract::AbstractTuple>(AnfUtils::VisitKernel(cnode, item_idx).first->abstract());
      cnode->set_abstract(abs_tuple->elements()[item_idx]);
      changed = true;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
