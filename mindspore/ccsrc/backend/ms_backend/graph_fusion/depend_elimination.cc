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

#include "backend/ms_backend/graph_fusion/depend_elimination.h"
#include "primitive/framework_ops.h"
#include "primitive/array_ops.h"
#include "primitive/math_ops.h"
#include "primitive/nn_optimizer_ops.h"
#include "include/utils/utils.h"
#include "include/utils/anfalgo.h"
#include "ir/graph_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

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
        // Optimize cases like %1 = Depend(%0, %0). This depend statement is necessary in frontend, but not here.
        (void)mng->Replace(node, inputs[kRealInputIndexInDepend]);
        MS_LOG(INFO) << "Depend node has been replaced by " << inputs[kRealInputIndexInDepend];
      }
    }
  }
  return true;
}

const BaseRef GeneratedDependElimination::DefinePattern() const {
  auto reducesum_node = VectorRef({prim::kPrimReduceSum, input2_, input3_});
  auto assign_node = VectorRef({prim::kPrimAssign, input1_, reducesum_node});
  auto depend_node = VectorRef({prim::kPrimDepend, input1_, assign_node});
  return depend_node;
}

const AnfNodePtr GeneratedDependElimination::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  constexpr size_t kGraphNodeNum = 2;
  auto input1 = utils::cast<AnfNodePtr>((*equiv)[input1_]);
  if (common::AnfAlgo::IsGraphKernel(input1)) {
    auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(input1);
    auto sub_graph_nodes = sub_graph->GetOrderedCnodes();
    if (sub_graph_nodes.size() == kGraphNodeNum) {
      auto node_prim = GetCNodePrimitive(sub_graph_nodes.front());
      MS_EXCEPTION_IF_NULL(node_prim);
      if (node_prim->name() == prim::kPrimBroadcastTo->name()) {
        auto cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        auto assign_node = cnode->input(kIndex2);
        MS_EXCEPTION_IF_NULL(assign_node);
        auto assign_cnode = assign_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(assign_cnode);
        auto reducesum_node = assign_cnode->input(kIndex2);
        return reducesum_node;
      }
    }
  }
  return nullptr;
}
}  // namespace mindspore::graphkernel
