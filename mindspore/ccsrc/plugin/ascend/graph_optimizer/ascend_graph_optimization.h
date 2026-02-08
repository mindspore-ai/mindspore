/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_GRAPH_OPTIMIZATION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_GRAPH_OPTIMIZATION_H_
#include <vector>
#include <set>
#include <memory>
#include <tuple>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
namespace mindspore {
namespace device {
namespace ascend {
class AscendGraphOptimization {
 public:
  static AscendGraphOptimization &GetInstance() {
    static AscendGraphOptimization instance;
    return instance;
  }
  void OptimizeACLGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo);
  void OptimizeACLGraphAfterKernelSelect(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo);
  void OptimizeACLGraphAfterInline(const KernelGraphPtr &graph);
  void UnifyMindIR(const KernelGraphPtr &graph);
  void AscendUnifyMindIR(const KernelGraphPtr &graph) const;
  void OptimizeACLGraphAfterCreateKernel(const KernelGraphPtr &graph);

 private:
  AscendGraphOptimization() {}
  ~AscendGraphOptimization() = default;
  AscendGraphOptimization(const AscendGraphOptimization &) = delete;
  AscendGraphOptimization &operator=(const AscendGraphOptimization &) = delete;
};

}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_GRAPH_OPTIMIZATION_H_
