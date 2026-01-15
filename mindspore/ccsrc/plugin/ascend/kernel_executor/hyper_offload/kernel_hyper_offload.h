/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_H_

#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "utils/hyper_offload/hyper_offload.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace hyper_offload {
using utils::hyper_offload::CollectAllNodeUsers;
using utils::hyper_offload::HyperOffloadConfig;
using utils::hyper_offload::HyperOffloadInput;
using utils::hyper_offload::IsD2HNode;
using utils::hyper_offload::IsH2DNode;
using utils::hyper_offload::IsViewOrInplaceNode;

struct ConditionSwitchInfo {
  CNodePtr switch_cnode;
  std::string true_subgraph;
  std::string false_subgraph;
  size_t true_branch_node_num;
  size_t false_branch_node_num;

  ConditionSwitchInfo(const CNodePtr &switch_cnode, const std::string &true_subgraph_name,
                      const std::string &false_subgraph_name)
      : switch_cnode(switch_cnode),
        true_subgraph(true_subgraph_name),
        false_subgraph(false_subgraph_name),
        true_branch_node_num(0),
        false_branch_node_num(0) {}
};

class KernelHyperOffloadOptimizer {
 public:
  void Run(const KernelGraphPtr &kernel_graph);
  void AddEventToHyperOffloadOps(const KernelGraphPtr &kernel_graph);

 private:
  static bool EnableHyperOffloadForGraph(const KernelGraphPtr &kernel_graph);
  static bool EnableHyperOffloadParameterForGraph(const KernelGraphPtr &kernel_graph);
  static bool EnableHyperOffloadActivationForGraph(const KernelGraphPtr &kernel_graph);
  static HyperOffloadInput GenerateHyperOffloadInput(const KernelGraphPtr &kernel_graph);
  static void UpdateRefMapForHyperOffloadInput(const KernelGraphPtr &kernel_graph);
  void ApplyHyperOffloadOptimizer(const KernelGraphPtr &kernel_graph);
  std::vector<mindspore::utils::hyper_offload::EventPlan> events_;
};

void ReorderControlFlowNodes(const KernelGraphPtr &kernel_graph);
}  // namespace hyper_offload
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_H_
