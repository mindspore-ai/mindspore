/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_INLINE_CONTROL_FLOW_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_INLINE_CONTROL_FLOW_SCHEDULER_H_

#include "runtime/graph_scheduler/actor/actor_set.h"

namespace mindspore {
namespace runtime {
class InlineControlFlowScheduler {
 public:
  InlineControlFlowScheduler() = default;
  ~InlineControlFlowScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(InlineControlFlowScheduler);

  // Link control arrows and fix the member variables for condition actors.
  void Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info);

 private:
  void LinkControlArrowByExecutionOrder(const KernelGraphPtr &graph, const GraphCompilerInfo &graph_compiler_info);
  // Fix the member variables for condition actors.
  void HandleConditionSwitchActor(const KernelActorPtr &kernel_actor);
  void HandleConditionGatherActor(const KernelActorPtr &kernel_actor);

  // Init the output branch info for condition actor.
  // For condition switch actor, the output arrow include all the output branch and should be distinguished.
  void InitOutputBranchInfoForConditionSwitchActor(ConditionSwitchActor *const condition_switch_actor,
                                                   const KernelGraphPtr &kernel_graph);
  void InitOutputControlBranchInfoForConditionSwitchActor(ConditionSwitchActor *const condition_switch_actor,
                                                          const KernelGraphPtr &kernel_graph);
  void InitOutputDataBranchInfoForConditionSwitchActor(ConditionSwitchActor *const condition_switch_actor,
                                                       const KernelGraphPtr &kernel_graph);
  void InitInputBranchInfoForConditionGatherActor(ConditionGatherActor *const condition_gather_actor,
                                                  const KernelGraphPtr &kernel_graph);
  void InitInputDataBranchInfoForConditionGatherActor(ConditionGatherActor *const condition_gather_actor,
                                                      const KernelGraphPtr &kernel_graph);
  void InitInputControlBranchInfoForConditionGatherActor(ConditionGatherActor *const condition_gather_actor,
                                                         const KernelGraphPtr &kernel_graph);

  // Fix ref count for condition actors.
  // In condition switch actor, the ref count of actor should be change to total num for both branch.
  // In condition gather actor, the ref count of gather input should add the ref count of gather output.
  // The ref count of ref node should be add to the input of condition actor.
  void FixRefCountByConditionSwitchActor(ConditionSwitchActor *const condition_switch_actor,
                                         const KernelGraphPtr &kernel_graph);
  void FixRefCountByKernelGraphRefMap(ConditionSwitchActor *const condition_switch_actor,
                                      const KernelGraphPtr &kernel_graph);
  void FixRefCountByConditionGatherActor(ConditionGatherActor *const condition_gather_actor,
                                         const KernelGraphPtr &kernel_graph);
  void FixRefCountByKernelGraphRefMap(ConditionGatherActor *const condition_gather_actor,
                                      const KernelGraphPtr &kernel_graph);

  std::string GetBranchNameByConditionGatherActor(KernelActor *condition_switch_actor,
                                                  KernelActor *condition_gather_actor, DataArrow *data_arrow,
                                                  const KernelGraphPtr &kernel_graph);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_INLINE_CONTROL_FLOW_SCHEDULER_H_
