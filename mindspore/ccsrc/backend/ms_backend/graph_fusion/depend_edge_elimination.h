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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_DEPEND_EDGE_ELIMINATION_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_DEPEND_EDGE_ELIMINATION_H_

#include "include/backend/common/pass_manager/pass.h"

namespace mindspore::graphkernel {
/**
 * @brief Eliminate redundant outputs from fused MatMul_AssignAdd operators by merging Depend edges
 * @details This pass optimizes cases where fused MatMul_AssignAdd operators produce multiple outputs
 *          that are consumed by both Depend operations and subsequent operators. By restructuring
 *          the dependency edges, we can reduce the number of subgraph outputs while maintaining
 *          execution order constraints.
 *
 * @example
 * // Original pattern:
 * sub_graph {
 *   %0 = MatMul_AssignAdd(p0, p1)  // Two outputs: [output_to_depend, output_to_next_op]
 *   return %0.0, %0.1
 * }
 * main_graph {
 *   %0 = sub_graph()
 *   %1 = tuple_getitem(%0, 0)  // output_to_depend
 *   %2 = tuple_getitem(%0, 1)  // output_to_next_op
 *   %3 = Depend(input, %2)  // Explicit dependency
 *   %4 = NextOp(%3)
 * }
 *
 * // Optimized pattern:
 * sub_graph {
 *   %0 = MatMul_AssignAdd(p0, p1)
 *   return %0  // Single output
 * }
 * main_graph {
 *   %0 = sub_graph()
 *   %1 = Depend(input, %0)
 *   %2 = NextOp(%0)
 * }
 */
class DependEdgeElimination : public opt::Pass {
 public:
  DependEdgeElimination() : Pass("depend_edge_elimination") {}
  ~DependEdgeElimination() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_DEPEND_EDGE_ELIMINATION_H_
