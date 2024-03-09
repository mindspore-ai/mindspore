/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/add_ref_pair.h"

#include <utility>
#include "ir/anf.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "kernel/framework_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"

namespace mindspore::graphkernel {
bool AddRefPair::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;

  auto todos = TopoSort(func_graph->get_return());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto node : todos) {
    if (!common::AnfAlgo::IsGraphKernel(node)) {
      continue;
    }
    auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(sub_graph);

    AnfNodePtrList output_list;
    kernel::GetFuncGraphOutputNodes(sub_graph, &output_list);
    auto graph_inputs = sub_graph->parameters();
    for (size_t i = 0; i < output_list.size(); ++i) {
      auto &out = output_list[i];
      auto out_cnode = out->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(out_cnode);
      if (GetCNodePrimitive(out_cnode)->name() == prim::kPrimAssign->name()) {
        auto iter = std::find(graph_inputs.begin(), graph_inputs.end(), out_cnode->input(kIndex1));
        if (iter == graph_inputs.end()) {
          MS_LOG(EXCEPTION) << out_cnode->fullname_with_scope() << " first input isn't parameter.";
        }
        auto input_idx = std::distance(graph_inputs.begin(), iter);
        auto origin_pair = common::AnfAlgo::GetPrevNodeOutput(node, input_idx, true);
        // record the ref_pair
        session::AnfWithOutIndex final_pair = std::make_pair(node, i);
        if (kernel_graph->IsInRefOutputMap(final_pair)) {
          MS_LOG(INTERNAL_EXCEPTION) << "Ref_pair is already in ref map, node is " << node->fullname_with_scope()
                                     << ", index is " << i;
        }
        MS_LOG(DEBUG) << "Add Ref pair, final {node ptr " << final_pair.first.get() << " , info is "
                      << final_pair.first->fullname_with_scope() << " , index is " << final_pair.second
                      << "}, origin {node ptr " << origin_pair.first.get() << ", info is "
                      << origin_pair.first->fullname_with_scope() << " : index " << origin_pair.second << "}";
        kernel_graph->AddRefCorrespondPairs(final_pair, origin_pair);
        changed = true;
      }
    }
  }

  return changed;
}
}  // namespace mindspore::graphkernel
