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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_SCHEDULER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <queue>
#include "backend/ms_backend/runtime/actors/base/actor_set.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_compiler.h"

namespace mindspore {
namespace runtime {
class ControlNodeScheduler {
 public:
  ControlNodeScheduler() = default;
  ~ControlNodeScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(ControlNodeScheduler);

  // Transform the control nodes to control actors.
  ControlActorSetPtr Build(const GraphCompilerInfo &graph_compiler_info, const AID &memory_manager_aid);
  // Link control actors.
  void Link(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) const;

  void BuildGraphParameterStoreForControlNode(const GraphCompilerInfo &graph_compiler_info,
                                              const AID &memory_manager_aid) const;

  void BuildDataSourceActorForControlNode(const GraphCompilerInfo &graph_compiler_info,
                                          const HostTensorQueuePtr &host_queue,
                                          const HostQueueDSActorPtr &host_queue_ds_actor, const AID &memory_manager_aid,
                                          std::vector<DataSourceActorPtr> *data_source_actors) const;

  // The control flow actor will generate some data in the loop body execution, so need clear on the end of execution.
  void ClearActorData(const ControlActorSet *control_actor_set) const;
  void ClearExitActorDeviceTensors(std::vector<ExitActorPtr> exit_actors) const;
  void Optimize(const ActorSetPtr &actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void DumpFormatControlActorSet(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                                 const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>,
                                                session::KernelWithIndexCmp> &graph_output_to_actor,
                                 std::ofstream &ofs);

 private:
  // Interface to create control actors.
  std::vector<SwitchActorPtr> BuildSwitchActor(const GraphCompilerInfo &graph_compiler_info) const;
  std::vector<GatherActorPtr> BuildGatherActor(const GraphCompilerInfo &graph_compiler_info) const;
  std::vector<EntranceActorPtr> BuildEntranceActor(const GraphCompilerInfo &graph_compiler_info) const;
  std::vector<ExitActorPtr> BuildExitActor(const GraphCompilerInfo &graph_compiler_info) const;
  std::vector<StackActorPtr> BuildStackActor(const GraphCompilerInfo &graph_compiler_info) const;
  void BuildStackActorForControlNode(const GraphCompilerInfo &graph_compiler_info,
                                     std::vector<StackActorPtr> *const stack_actors) const;
  // Interface to link control actors.
  void LinkControlArrowForControlActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void LinkControlArrowForEntranceActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void LinkBranchIDArrowForControlActor(ControlActorSet *const control_actor_set) const;
  // Link all arrows between control actors.
  void LinkArrowForControlActor(ControlActorSet *const control_actor_set,
                                const GraphCompilerInfo &graph_compiler_info) const;
  void LinkArrowbyFormalParameter(ControlActor *const to_actor, const KernelWithIndex &from_node_with_index,
                                  const KernelWithIndex &to_node_with_index,
                                  const GraphCompilerInfo &graph_compiler_info) const;
  void LinkArrowByCallNode(const AnfNodePtr &call_node, ControlActor *const to_actor,
                           const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index,
                           const GraphCompilerInfo &graph_compiler_info) const;
  void LinkArrowByKernel(const AnfNodePtr &kernel, ControlActor *const to_actor,
                         const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index,
                         const GraphCompilerInfo &graph_compiler_info) const;
  void LinkArrowByParameter(const AnfNodePtr &parameter, ControlActor *const to_actor,
                            const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index,
                            const ControlNodeParserPtr &parser) const;
  void LinkArrowByValueNode(const AnfNodePtr &value_node, ControlActor *const to_actor, size_t from_index,
                            size_t to_index) const;
  // Link arrow from stack actor to control actor.
  void LinkArrowFromStackActor(StackActor *stack_actor, ControlActor *to_actor,
                               const GraphCompilerInfo &graph_compiler_info) const;

  // Link data arrow between control actor and actor in frame, including kernel actor, output actor, data source actor.
  void LinkDataArrowForKernelActor(const GraphCompilerInfo &graph_compiler_info) const;
  void LinkDataArrowByKernelGraph(const KernelGraphPtr &graph, ControlActor *const entrance_actor,
                                  const ControlNodeParserPtr &parser) const;
  void LinkDataArrowByKernelGraphInSinkMode(const KernelGraphPtr &graph, ControlActor *const from_actor,
                                            const ControlNodeParserPtr &parser) const;
  void LinkArrowForRootGraphEntranceActor(const ActorSet *actor_set,
                                          const GraphCompilerInfo &graph_compiler_info) const;
  void LinkControlArrowForLoopCountActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void LinkDataArrowForOutputActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void LinkControlArrowForKernelActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void LinkOutputControlArrowForActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void LinkControlArrowByKernelGraphGroup(const GraphCompilerInfo &graph_compiler_info) const;
  void LinkControlArrowByAutoMonad(ControlActor *to_actor, const ControlNodeParserPtr &parser,
                                   const std::set<AnfNodePtr> &depend_nodes) const;

  // Add time summary info for counting the execution time between two actors.
  void SetTimeSummaryForControlActor(const GraphCompilerInfo &graph_compiler_info) const;
  bool IsNoInputActor(const ControlActor *control_actor) const;
  void CollectDynamicLenIndexForArgment(const GraphCompilerInfo &graph_compiler_info) const;

  void LinkControlArrowBySegmentTopoSort(const GraphCompilerInfo &graph_compiler_info) const;
  void OptimizeBranchIdArrow(const ActorSetPtr &actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void OptimizeDynamicRefCountForEntranceActor(const ActorSetPtr &actor_set) const;
  void OptimizeDynamicRefCountForStackActor(const ActorSetPtr &actor_set) const;
  void OptimizeDynamicRefCountForGatherActor(const ActorSetPtr &actor_set,
                                             const GraphCompilerInfo &graph_compiler_info) const;

  void CollectIgnoreIndexForEntranceActor(std::set<int> *ignore_index, const EntranceActorPtr &entrance_actor) const;

  bool CheckIsValidArgIndex(size_t index, const EntranceActorPtr &entrance_actor, const ControlActor *gather_actor,
                            const FuncGraphPtr &func_graph, const CNodePtr &partial_cnode, size_t *to_index) const;
  std::vector<std::string> GetInputAids(AbstractActor *const actor, const ControlNodeParserPtr &parser,
                                        const std::unordered_map<std::string, std::string> &exit_to_gather,
                                        const FuncGraphPtr &func_graph);
  void DumpControlActorInfo(const ExitActorPtr &exit_actor, const ControlNodeParserPtr &parser,
                            const std::unordered_map<std::string, std::string> &exit_to_gather,
                            const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>,
                                           session::KernelWithIndexCmp> &graph_output_to_actor,
                            std::ofstream &ofs);
  // The id of memory manager actor.
  AID memory_manager_aid_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_SCHEDULER_H_
