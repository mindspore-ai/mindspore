/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/tensor_promotion.h"
#include <vector>
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
bool TensorPromotion::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto todos = TopoSort(func_graph->get_return());

  bool changed = false;
  for (auto iter = todos.crbegin(); iter != todos.crend(); ++iter) {
    auto node = *iter;
    if (!AnfAlgo::IsGraphKernel(node)) {
      continue;
    }
    auto args = node->cast<CNodePtr>()->inputs();
    auto fg = GetValueNode<FuncGraphPtr>(args[kAnfPrimitiveIndex]);
    if (!ConvertNonscalarTensorToParameter(fg, &args)) {
      continue;
    }
    AnfNodePtrList inputs, outputs;
    inputs.insert(inputs.end(), args.begin() + 1, args.end());
    kernel::GetFuncGraphOutputNodes(fg, &outputs);
    auto new_cnode = CreateNewFuseCNode(func_graph, fg, inputs, outputs);
    SetNewKernelInfo(new_cnode, fg, inputs, outputs, AnfAlgo::GetProcessor(node));
    mng->Replace(node, new_cnode);
    changed = true;
  }

  return changed;
}
}  // namespace opt
}  // namespace mindspore
