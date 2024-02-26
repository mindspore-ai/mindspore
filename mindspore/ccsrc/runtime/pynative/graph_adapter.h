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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_GRAPH_ADAPTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_GRAPH_ADAPTER_H_

#include <vector>
#include <set>
#include <unordered_map>
#include "include/backend/kernel_graph.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/graph_compiler.h"

namespace mindspore::pynative {
using GraphCompilerInfo = runtime::GraphCompilerInfo;
class GraphAdapter {
 public:
  void UpdateForwardOutputInBpropGraph(const KernelGraphPtr &graph, const device::DeviceContext *device_context,
                                       bool no_control_flow);
  void GenerateBackoffValueNodeOwners(const KernelGraphPtr &graph);
  static void ReplaceGraphParameterProperties(const KernelGraphPtr &graph,
                                              const std::vector<tensor::TensorPtr> &input_tensors,
                                              const device::DeviceContext *device_context);
  static void GenerateRefCountForBpropValueNode(const KernelGraphPtr &graph);
  static void ClearForwardOutputValueNodeDeviceAddress(const KernelGraphPtr &graph,
                                                       const device::DeviceContext *device_context);
  static void RemoveUnusedValueNodes(const KernelGraphPtr &graph);
  static void HandleHeterogeneousTensors(const std::vector<std::vector<tensor::TensorPtr>> &tensors,
                                         const std::vector<device::DeviceContext *> &device_contexts);
  static bool IsPynativeGeGraphSink(const GraphCompilerInfo &graph_compiler_info);
  static bool IsPynativeGeGraphSink(const FuncGraphPtr &func_graph);
  static bool PyNativeEnableTaskSink(const FuncGraphPtr &func_graph);
  static bool IsAutoParallel();
  static void UpdateDynamicValueNodeAbstract(const KernelGraphPtr &graph);
  static void SensTensorToDevice(const KernelGraphPtr &graph, const device::DeviceContext *device_context);

 private:
  void HandleBackoffValueNode(const ValueNodePtr &value_node, const AnfNodePtr &front_node,
                              const DeviceContext *device_context) const;
  // Each backend has an independent map.
  // The map will be destroyed when the backend object is destroyed.
  std::unordered_map<AnfNode *, std::set<CNodePtr>> node_to_backoff_kernels_;
};
}  // namespace mindspore::pynative
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_GRAPH_ADAPTER_H_
