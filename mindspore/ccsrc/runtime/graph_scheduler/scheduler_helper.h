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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_SCHEDULER_HELPER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_SCHEDULER_HELPER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <set>
#include <algorithm>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "runtime/graph_scheduler/actor/actor_set.h"

namespace mindspore {
namespace runtime {
class SchedulerHelper {
 public:
  // Convert the actors vector by the actor set.
  static std::vector<AbstractActorPtr> CollectActors(const ActorSet *actor_set);

  // Judge the input node whether need the control arrow.
  static bool HasMonadControl(const AnfNodePtr &input_node, const KernelGraphPtr &graph);

  static void AddDeviceTensorStore(const AnfNode *anf_node, const DeviceTensorPtr &device_tensor);

  // Add the arrow between from actor and to actor.
  static void AddDataArrow(AbstractActor *const from_actor, AbstractActor *const to_actor, size_t from_output_index,
                           size_t to_input_index, const AnfNodePtr &from_kernel = nullptr);
  static void AddResultArrow(AbstractActor *const from_actor, OutputActor *const to_actor,
                             const AnfNodePtr &from_kernel, size_t from_output_index, size_t output_position);
  static void AddControlArrow(AbstractActor *const from_actor, AbstractActor *const to_actor);

  // Add the arrow for control actor.
  static void AddPartialArrow(ControlActor *const from_actor, ControlActor *const to_actor, size_t from_index,
                              size_t to_index);
  static void AddBranchIDArrow(ControlActor *const from_actor, ControlActor *const to_actor);
  // Body control arrow is only exists to entrance actor..
  static void AddLoopBodyControlArrow(AbstractActor *from_actor, EntranceActor *to_actor);
  // Data arrow with branch id is only exists from gather actor to entrance actor.
  static void AddDataWithBranchIDArrow(GatherActor *const gather_actor, const EntranceActor *entrance_actor,
                                       const FuncGraphPtr &func_graph);
  // Since the output of exit actor has branches, it needs to be based on a dedicated interface.
  static void AddDataArrowForExitActor(ExitActor *const exit_actor, AbstractActor *const to_actor, size_t from_index,
                                       size_t to_index, int branch_id);
  static void AddPartialArrowForExitActor(ExitActor *const exit_actor, ControlActor *const to_actor, size_t from_index,
                                          size_t to_index, int branch_id);
  static void AddControlArrowForExitActor(ExitActor *from_actor, AbstractActor *to_actor, int branch_id);

  // Fill the device tensors of backend input nodes corresponding to ref formal parameters.
  static void AddFormalParameterDeviceTensor(ControlActor *const from_actor, size_t from_index,
                                             const AnfNodePtr &input_node, const KernelGraphPtr &graph);

  // Convert the invalid data arrow to control arrow.
  static void ConvertDataArrowToControlArrow(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                             const DataArrowPtr &data_arrow, size_t data_arrow_index);

  // Fuse the data arrows to batch data arrow for the same destination actor.
  static void FuseDataArrowsToBatchDataArrow(AbstractActor *const actor);

  // The interface of fusing the actors to a fusion actor.
  static void AddDependency(AbstractActor *const actor, const AbstractActor *dependent_actor);
  static bool CheckDependency(const std::vector<AbstractActorPtr> &output_actors);
  static FusionActorPtr BuildFusionActor(const std::vector<AbstractActorPtr> &actors);
  static void AddArrowForFusionActor(FusionActor *fusion_actor);

  // The interface of integration of dynamic and static memory.
  static void AddMemorySign(AbstractActor *const from_actor, AbstractActor *const to_actor);
  static KernelGraphPtr FecthKernelGraphByActor(AbstractActor *const actor);
  // Add the memory alloc sign for the head kernel actor of graph.
  static void AddMemoryAllocSign(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                 const KernelGraphPtr &to_graph);
  // Add the memory free sign for the tail kernel actor of graph.
  static void AddMemoryFreeSign(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                const KernelGraphPtr &from_graph);
  static void AddSomasInfo(AbstractActor *const actor);

  // Check whether the actor set is valid.
  static void CheckActorValid(const ActorSet *actor_set);

  static void DumpActorSet(const ActorSet *actor_set, std::ofstream &ofs);

  static size_t fusion_actor_index_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_
