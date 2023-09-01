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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_ELIMINATION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_ELIMINATION_H_

#include <memory>

#include "include/backend/optimizer/pass.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::graphkernel {
// Optimize cases like %1 = Depend(%0, %0). This depend statement is necessary in frontend, but not here.
class DependElimination : public opt::Pass {
 public:
  DependElimination() : Pass("depend_elimination") {}
  ~DependElimination() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};

/**
 * @brief the atomic add kernel compilation on Ascend or by AKG_V2 is failed, this kernel will be inlined to main graph.
 * Then there will be unnecessary nodes in the main graph. This pass is used to eliminate these unnecessary nodes.
 * @example
 * Before atomic clean pass, the graph is:
 * main(p0) {
 *   %1 = ReduceSum(p0)
 * }
 * After atomic clean pass, the graph should look like this:
 * main(p0) {
 *   %0 = call fg0()
 *   %1 = call fg1(%0, p0)
 *   %2 = Depend(%0, %1)
 *   %3 = op(%2)
 * }
 * fg0() {
 *   %0 = BroadcastTo(0)
 *   return %0
 * }
 * fg1(p0, p1) {
 *   %0 = ReduceSum(p1)
 *   %1 = Assign(p0, p1)
 *   return %0
 * }
 * If the compilation of graph kernel fg1 is failed, it will be inlined. Then there will be redundant nodes fg0,
 * Assign and Depend in this graph. For performance, we need to eliminate this generated Depend node, (fg0 and Assign
 * will have no users and be deleted), then the graph will be the same as the graph before atomic clean pass.
 */
class GeneratedDependElimination : public opt::PatternProcessPass {
 public:
  explicit GeneratedDependElimination(bool multigraph = true)
      : PatternProcessPass("generated_depend_elimination", multigraph),
        input1_{std::make_shared<Var>()},
        input2_{std::make_shared<Var>()},
        input3_{std::make_shared<Var>()} {};
  ~GeneratedDependElimination() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  VarPtr input1_;
  VarPtr input2_;
  VarPtr input3_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_ELIMINATION_H_
