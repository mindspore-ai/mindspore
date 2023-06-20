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
#include "tools/graph_kernel/converter/split_umonad.h"

#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"

namespace mindspore::graphkernel {
/*
 *  %1 = Assign(param, %0, UMonad)
 *  =================>
 *  %1 = Assign(param, %0)
 *  %2 = Depend(%1, UMonad)
 * */
bool SplitAssign::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (const auto &node : todos) {
    if (node == nullptr) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !IsPrimitiveCNode(cnode, prim::kPrimAssign)) {
      continue;
    }
    constexpr size_t umonad_idx = 3;
    if (cnode->inputs().size() != umonad_idx + 1) {
      continue;
    }
    auto umonad = cnode->input(umonad_idx);
    if (!HasAbstractUMonad(umonad)) {
      continue;
    }
    AnfNodePtrList new_inputs(cnode->inputs().begin(), cnode->inputs().begin() + umonad_idx);
    cnode->set_inputs(new_inputs);
    auto depend_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimDepend), cnode, umonad});
    depend_cnode->set_abstract(node->abstract()->Clone());
    (void)mng->Replace(node, depend_cnode);
    changed = true;
  }
  return changed;
}
}  // namespace mindspore::graphkernel
