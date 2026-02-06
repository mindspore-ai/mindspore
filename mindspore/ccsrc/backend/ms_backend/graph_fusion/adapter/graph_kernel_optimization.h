/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_OPTIMIZATION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_OPTIMIZATION_H_

#include <vector>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/common/pass_manager/pass_manager.h"

namespace mindspore::graphkernel {
using opt::PassManagerPtr;
class GraphKernelOptimizer {
 public:
  void Run(const KernelGraphPtr &kernel_graph);
  void RunKernelPacket(const KernelGraphPtr &kernel_graph);

 private:
  void Init() const;
  // Pre-process
  PassManagerPtr PreProcess() const;
  // Cluster kernels
  PassManagerPtr Cluster() const;
  // High level optimize 1
  PassManagerPtr HighLevelOpt1() const;
  // Split kernels
  PassManagerPtr Split() const;
  // High level optimize 2
  PassManagerPtr HighLevelOpt2() const;
  // Combine kernels
  PassManagerPtr Combine() const;
  // Build kernels
  PassManagerPtr Build() const;
  // Post-process
  PassManagerPtr PostProcess() const;
  // Pack a device op with input host ops
  PassManagerPtr KernelPacket() const;

  bool is_gpu{false};
  bool is_ascend{false};
  bool is_cpu{false};
  bool is_dvm{false};
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_OPTIMIZATION_H_
