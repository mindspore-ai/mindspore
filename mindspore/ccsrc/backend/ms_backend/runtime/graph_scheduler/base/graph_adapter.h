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
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "backend/ms_backend/runtime/actors/base/actor_set.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_compiler.h"

namespace mindspore::pynative {
using GraphCompilerInfo = runtime::GraphCompilerInfo;
using ActorSet = runtime::ActorSet;
class GraphAdapter {
 public:
  void UpdateForwardOutputInBpropGraph(const KernelGraphPtr &graph, const device::DeviceContext *device_context,
                                       bool no_control_flow);
  void GenerateBackoffValueNodeOwners(const KernelGraphPtr &graph);
  static void RemoveUnusedValueNodes(const KernelGraphPtr &graph);
  static void HandleHeterogeneousTensors(const std::vector<std::vector<tensor::TensorPtr>> &tensors,
                                         const std::vector<device::DeviceContext *> &device_contexts,
                                         ActorSet *actor_set);
  static bool IsPynativeGeGraphSink(const GraphCompilerInfo &graph_compiler_info);
  static bool IsPynativeGeGraphSink(const FuncGraphPtr &func_graph);
  static bool IsAutoParallel();
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
