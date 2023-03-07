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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MINDSPORE_ASCEND_GRAPH_OPTIMIZATION_H
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MINDSPORE_ASCEND_GRAPH_OPTIMIZATION_H

#include <vector>
#include <set>
#include <memory>
#include "include/backend/kernel_graph.h"
#include "backend/common/optimizer/graph_optimizer.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendGraphOptimization {
 public:
  static AscendGraphOptimization &GetInstance() {
    static AscendGraphOptimization instance;
    return instance;
  }

  void OptimizeGraph(const KernelGraphPtr &graph);
  void OptimizeSingleOpGraph(const KernelGraphPtr &graph);
  void UnifyMindIR(const KernelGraphPtr &graph);
  void AddUnifyMindIRPass(const std::shared_ptr<opt::GraphOptimizer> &opt) const;
  void OpAdaptation(const KernelGraphPtr &graph);
  void Reset();

 private:
  AscendGraphOptimization() { graph_manager_ = MakeManager(); }
  ~AscendGraphOptimization() = default;
  AscendGraphOptimization(const AscendGraphOptimization &) = delete;
  AscendGraphOptimization &operator=(const AscendGraphOptimization &) = delete;

  // Select the matching backend kernels according to the data type and format of input and output for all
  // execution operators, and set final device data type and format information for backend kernels, device
  // data type and format which replace original data type and format will use for executing kernels.
  void SetOperatorInfo(const KernelGraphPtr &graph);
  // Graph Optimized level-2 interface
  void OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph);
  void OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph);
  void OptimizeExecutionOrder(const KernelGraphPtr &graph);
  void InlineSubGraph(const KernelGraphPtr &graph);
  void PostOptimization(const KernelGraphPtr &graph) const;

  // Graph Optimized level-3 interface
  void CommOpReuse(const KernelGraphPtr &graph) const;
  void IRFusionOptimization(const KernelGraphPtr &graph);
  void UpdateRefOutputMap(const KernelGraphPtr &graph);
  void AddGraphToManager(const NotNull<KernelGraphPtr> graph, const NotNull<FuncGraphManagerPtr> manager,
                         bool is_root = true);
  void SelectKernel(const KernelGraphPtr &graph);
  void AscendInsertTypeTransformOps(const KernelGraphPtr &kernel_graph);
  void RecurseSelectKernelInfo(const KernelGraphPtr &graph);
  void HardWareOptimization(const KernelGraphPtr &graph);
  void HandleControlFlow(const NotNull<KernelGraphPtr> graph) const;
  void RootGraphExecutorValidate(const NotNull<KernelGraphPtr> graph);

  void GetAllGraphs(const KernelGraphPtr &root_graph);
  void CheckControlFlowDynamicShape(const KernelGraphPtr &root_graph);
  void AclOpOptimize(const KernelGraphPtr &graph);

  // Manager for the optimized graphs
  FuncGraphManagerPtr graph_manager_;
  // Number of operators whose precision changes after select kernel
  size_t raise_precision_count_{0};
  size_t reduce_precision_count_{0};
  // The graphs has been traversed when the graph id traversed recursively.
  // Note: Please clean the set before each use.
  std::set<KernelGraphPtr> memo_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_ASCEND_GRAPH_OPTIMIZATION_H
