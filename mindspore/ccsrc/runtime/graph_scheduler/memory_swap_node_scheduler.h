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
#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_SWAP_NODE_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_SWAP_NODE_SCHEDULER_H_

#include <vector>
#include <map>
#include <utility>
#include <memory>

#include "utils/ms_utils.h"
#include "runtime/device/memory_offload_strategy.h"
#include "runtime/graph_scheduler/actor/memory/memory_swap_actor.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/graph_scheduler/actor/actor_set.h"

namespace mindspore {
namespace runtime {
using MemOffloadStrategyPtr = std::shared_ptr<device::MemOffloadStrategy<DeviceTensor *>>;
class MemorySwapNodeScheduler {
 public:
  MemorySwapNodeScheduler() = default;
  ~MemorySwapNodeScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(MemorySwapNodeScheduler);

  std::vector<std::vector<MemSwapActorPtr>> Build(const GraphCompilerInfo &graph_compiler_info,
                                                  const AID *recorder_aid);
  void Link(const GraphCompilerInfo &graph_compiler_info, ActorSet *actor_set) const;

 private:
  MemOffloadStrategyPtr GenMemOffloadStrategy(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                              const ControlNodeParserPtr &parser,
                                              std::map<DeviceTensor *, void *> *mem_allocated);

  device::GraphMemStatistic<DeviceTensor *> CollectMemStatistic(const KernelGraphPtr &graph,
                                                                const DeviceContext *device_context,
                                                                const ControlNodeParserPtr &parser);
  void CollectGraphInputMemStatistic(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                     const ControlNodeParserPtr &parser,
                                     device::GraphMemStatistic<DeviceTensor *> *statistic);
  void CollectKernelInputMemStatistic(size_t kernel_index, const KernelGraphPtr &graph,
                                      const DeviceContext *device_context,
                                      device::GraphMemStatistic<DeviceTensor *> *statistic,
                                      HashSet<const void *> *offload_conflict) const;
  void CollectKernelOutputMemStatistic(size_t kernel_index, const KernelGraphPtr &graph,
                                       device::GraphMemStatistic<DeviceTensor *> *statistic,
                                       HashSet<const void *> *offload_conflict) const;
  void CollectKernelWorkspaceMemStatistic(size_t kernel_index, const KernelGraphPtr &graph,
                                          device::GraphMemStatistic<DeviceTensor *> *statistic,
                                          HashSet<const void *> *offload_conflict) const;
  DeviceTensor *GetNodeOutputDeviceTensor(const AnfNodePtr &node, size_t output_idx,
                                          const mindspore::KernelGraphPtr &graph,
                                          const mindspore::device::DeviceContext *device_context) const;

  bool MockStrategy(const MemOffloadStrategyPtr &strategy, const DeviceContext *device_context, size_t execution_size,
                    std::map<DeviceTensor *, void *> *mem_allocated) const;

  std::vector<MemSwapActorPtr> GenSwapActorsForGraph(const KernelGraphPtr &graph, DeviceContext *device_context,
                                                     const MemOffloadStrategyPtr &strategy,
                                                     const ControlNodeParserPtr &parser);
  void FilterRealParamInPreEvent(const device::MemEventPtrList<DeviceTensor *> &pre_events,
                                 const mindspore::KernelGraphPtr &graph, const EntranceActor *entrance_actor,
                                 device::MemEventPtrList<DeviceTensor *> *swap_in_events,
                                 std::vector<size_t> *entrance_index) const;
  MemSwapActorPtr GenSwapInActor(const CNodePtr &kernel, DeviceContext *device_context,
                                 const device::MemEventPtrList<DeviceTensor *> &swap_events,
                                 const std::vector<device::ContinuousMemInfoPtr<DeviceTensor *>> &continuous_mem_info,
                                 size_t real_parameter_size) const;
  MemSwapActorPtr GenSwapOutActor(const CNodePtr &kernel, const device::MemEventPtrList<DeviceTensor *> &swap_events,
                                  const std::vector<bool> &swap_out_real_parameter_without_max_ref_count) const;
  void LinkControlArrowBySwapActor(const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                                   const std::vector<MemSwapActorPtr> &swap_actors) const;
  void LinkDataArrowForRealParameter() const;

  const AID *recorder_aid_;
  std::map<const DeviceTensor *, AnfNodePtr> formal_parameters_;
  std::map<std::shared_ptr<MemorySwapActor>, std::pair<EntranceActor *, std::vector<size_t>>> real_parameter_map_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_SWAP_NODE_SCHEDULER_H_
