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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_OPTIMIZE_ASSIGN_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_OPTIMIZE_ASSIGN_H_

#include <memory>
#include "include/backend/common/pass_manager/pass.h"

namespace mindspore::graphkernel {
/**
 * \brief If an Assign's source node was outputted with this Assign, the src-node can be removed from output list,
 * external users can use the dest-node under the premise of correct execution order.
 * \note
 *   1. Assign is always in output list. (links to external Depend node).
 *   2. Assign's dest-node should be a Parameter.
 *   3. Source node can not be the output of KernelGraph.
 *
 * \example
 * --- original graph ---
 * sub_fg(p1, p2):
 *   %0 = Add(p1, p2)
 *   %1 = Assign(p1, %0)
 *   %2 = make_tuple(%0, %1)
 *   return %2
 * main_kg():
 *   %0 = op1()
 *   %1 = call sub_fg(param1, %0)
 *   %2 = tuple_getitem(%1, 0)  // the output of Add node
 *   %3 = op2(%2)
 *   ...
 * --- after OptimizeAssign -->
 * main_kg():
 *   %0 = op1()
 *   %1 = call sub_fg(param1, %0)  // the subgraph is not changed in this pass.
 *   %2 = tuple_getitem(%1, 1) // the output of Assign node
 *   %3 = UpdateState(U, %2)
 *   %4 = Load(param1, %3)
 *   %5 = op2(%4)
 *   ...
 * --- after EliminateRedundantOutput -->
 * sub_fg(p1, p2):
 *   %0 = Add(p1, p2)
 *   %1 = Assign(p1, %0)
 *   return %1  // do not output the Add
 * main_kg():
 *   %0 = op1()
 *   %1 = call sub_fg(param1, %0)
 *   %2 = UpdateState(U, %1)
 *   %3 = Load(%0, %2)
 *   %4 = op2(%3)
 *   ...
 */
class OptimizeAssign : public opt::Pass {
 public:
  OptimizeAssign() : Pass("optimize_assign") {}
  ~OptimizeAssign() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
using OptimizeAssignPtr = std::shared_ptr<OptimizeAssign>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_OPTIMIZE_ASSIGN_H_
