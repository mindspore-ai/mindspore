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

#include "runtime/graph_scheduler/optimizer/multi_actor_fusion.h"
#include <vector>
#include "runtime/graph_scheduler/scheduler_helper.h"

namespace mindspore {
namespace runtime {
constexpr size_t kActorFusionMaxNum = 2;

void MultiActorFusion::Process(ActorSet *const actor_set, AbstractActor *const) {
  MS_EXCEPTION_IF_NULL(actor_set);
  if (actor_set->control_actors_ != nullptr) {
    return;
  }

  // Build all the fusion actors.
  std::vector<AbstractActorPtr> actors;
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    (void)actors.emplace_back(kernel_actor);
    if (actors.size() % kActorFusionMaxNum == 0) {
      auto fusion_actor = SchedulerHelper::BuildMultiActors(actors);
      (void)actor_set->fusion_actors_.emplace_back(fusion_actor);
      actors.clear();
    }
  }
  if (actors.size() > 1) {
    auto fusion_actor = SchedulerHelper::BuildMultiActors(actors);
    (void)actor_set->fusion_actors_.emplace_back(fusion_actor);
  }

  // Link fusion actor.
  for (auto &fusion_actor : actor_set->fusion_actors_) {
    SchedulerHelper::AddArrowForFusionActor(fusion_actor.get());
  }
}
}  // namespace runtime
}  // namespace mindspore
