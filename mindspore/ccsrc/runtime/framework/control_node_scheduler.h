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
#include "runtime/framework/actor/actor_set.h"
#include "runtime/framework/graph_compiler.h"

namespace mindspore {
namespace runtime {
class ControlNodeScheduler {
 public:
  ControlNodeScheduler() = default;
  ~ControlNodeScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(ControlNodeScheduler);

  // Transform the control nodes to control actors.
  ControlActorSetPtr Build(const GraphCompilerInfo &graph_compiler_info) { return nullptr; }
  // Link control actors.
  void Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {}

  bool CheckActorValid(const ControlActorSetPtr &control_actor_set);

 private:
  // Interface to create control actors.
  std::vector<SwitchActorPtr> BuildSwitchActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<GatherActorPtr> BuildGatherActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<EntranceActorPtr> BuildEntranceActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<ExitActorPtr> BuildExitActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<StackActorPtr> BuildStackActor(const GraphCompilerInfo &graph_compiler_info);

  // Interface to link control actors.
  void LinkControlArrowForControlActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkBranchIDArrowForControlActor(ControlActorSet *const control_actor_set);
  // Link all arrows between control actors.
  void LinkArrowForControlActor(ControlActorSet *const control_actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkArrowbyFormalParameter(ControlActor *const to_actor, const KernelWithIndex &from_node_with_index,
                                  const KernelWithIndex &to_node_with_index, const ControlNodeParserPtr &parser);

  // Link data arrow between control actor and actor in frame, including kernel actor, output actor, data source actor.
  void LinkDataArrowForKernelActor(const GraphCompilerInfo &graph_compiler_info);
  void LinkDataArrowForOutputActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkDataArrowForHostDSActor(const GraphCompilerInfo &graph_compiler_info);

  // Interface tool to link arrows between actors.
  void LinkControlArrow(AbstractActor *from_actor, AbstractActor *to_actor);
  // Data arrow with branch id is only exists from gather actor to entrance actor.
  void LinkDataWithBranchIDArrow(GatherActor *const gather_actor, EntranceActor *const entrance_actor,
                                 int from_branch_id);
  void LinkPartialArrow(ControlActor *const from_actor, ControlActor *const to_actor, size_t from_index,
                        size_t to_index);
  void LinkDataArrow(AbstractActor *const exit_actor, AbstractActor *const to_actor,
                     const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index);

  // Since the output of exit actor has branches, it needs to be based on a dedicated interface.
  void LinkControlArrowForExitActor(ExitActor *from_actor, AbstractActor *to_actor, int branch_id);
  void LinkDataArrowForExitActor(ExitActor *const exit_actor, AbstractActor *const to_actor,
                                 const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index,
                                 int branch_id);
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_SCHEDULER_H_
