/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_GRAPH_EXECUTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_GRAPH_EXECUTOR_H_

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendGraphExecutor : public GraphExecutor {
 public:
  AscendGraphExecutor() = default;
  ~AscendGraphExecutor() override = default;

  void Initialize();
  void Destroy();

  // Launch graph, device such as Ascend support the whole graph sink to the device executing.
  bool RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                std::vector<tensor::Tensor> *outputs, const std::map<string, string> &compile_options) override;

  // Adjust kernel graph before run graph, used in Graph Mode.
  void PreprocessBeforeRun(const KernelGraphPtr &graph) const;
  std::string GetRandomStatus(const std::vector<FuncGraphPtr> &graphs) override;

 private:
  // compile graph interface
  void UpdateExecOrder(const KernelGraphPtr &graph) const;
  void AllocateGraphMemory(const NotNull<KernelGraphPtr> &root_graph) const;
  void AssignInputMemory(const NotNull<KernelGraphPtr> &graph, NotNull<std::set<KernelGraphPtr> *> const memo) const;
  void LoadModel(const NotNull<KernelGraphPtr> &root_graph) const;

  // LaunchGraph interface
  bool ExecuteGraph(const KernelGraphPtr &graph) const;

  // Kernel Runtime  --- only for task sink
  AscendKernelRuntime *runtime_instance_{nullptr};
  std::shared_ptr<MemoryManager> mem_manager_{nullptr};

  // The ExecuteGraph is not thread safety specifically, it is not recommended that multiple threads access the same
  // func at the same time, so need the launch mutex when multiple threads launch the graph.
  mutable std::mutex launch_mutex_;
  // Using node to get its atomics
  mutable std::map<CNodePtr, std::vector<CNodePtr>> node_atomics_;
  AscendDeviceResManager *res_manager_{nullptr};
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_GRAPH_EXECUTOR_H_
