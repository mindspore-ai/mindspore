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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_OPTIMIZATION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_OPTIMIZATION_H_

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "backend/session/kernel_graph.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/common/pass_manager.h"

namespace mindspore {
namespace opt {
class GraphKernelOptimizer {
 public:
  void Run(const KernelGraphPtr &kernel_graph);

 private:
  // Pre-process
  PassManagerPtr PreProcess();
  // Cluster kernels
  PassManagerPtr Cluster();
  // High level optimize 1
  PassManagerPtr HighLevelOpt1();
  // Split kernels
  PassManagerPtr Split();
  // High level optimize 2
  PassManagerPtr HighLevelOpt2();
  // Combine kernels
  PassManagerPtr Combine();
  // Post-process
  PassManagerPtr PostProcess();

  bool is_gpu{false};
  bool is_ascend{false};
};

void GraphKernelOptimize(const KernelGraphPtr &kernel_graph);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_OPTIMIZATION_H_
