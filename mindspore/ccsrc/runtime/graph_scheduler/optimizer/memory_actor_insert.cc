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

#include "runtime/graph_scheduler/optimizer/memory_actor_insert.h"
#include <string>
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "runtime/graph_scheduler/actor/memory/memory_alloc_actor.h"
#include "runtime/graph_scheduler/actor/memory/memory_free_actor.h"

namespace mindspore {
namespace runtime {
bool MemoryActorInsert::MatchPattern(const AbstractActor *actor) const {
  MS_EXCEPTION_IF_NULL(actor);
  if ((actor->memory_alloc_insert_position() == nullptr) && (actor->memory_free_insert_position() == nullptr)) {
    return false;
  }
  return true;
}

void MemoryActorInsert::Process(ActorSet *const actor_set, AbstractActor *const actor) {
  MS_EXCEPTION_IF_NULL(actor);
  auto graph = SchedulerHelper::FecthKernelGraphByActor(actor);
  MS_EXCEPTION_IF_NULL(graph);
  auto somas_info = graph->MutableSomasInfo();
  MS_EXCEPTION_IF_NULL(somas_info);

  if (actor->memory_alloc_insert_position() != nullptr) {
    InsertMemoryAllocActor(actor_set, actor, somas_info);
  }

  if (actor->memory_free_insert_position() != nullptr) {
    InsertMemoryFreeActor(actor_set, actor, somas_info);
  }
}

void MemoryActorInsert::InsertMemoryAllocActor(ActorSet *const actor_set, AbstractActor *const actor,
                                               SomasInfo *somas_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(somas_info);

  std::string alloc_actor_name = "kernel_graph_" + std::to_string(somas_info->graph_id_) + kMemoryAllocActorNameSuffix;
  auto alloc_actor = FetchActor(alloc_actor_name);
  if (alloc_actor == nullptr) {
    // Build the memory alloc actor.
    auto &device_contexts = actor->device_contexts();
    MS_EXCEPTION_IF_CHECK_FAIL((!device_contexts.empty()), "The device context doesn't exist.");
    auto memory_aware_actor = dynamic_cast<MemoryAwareActor *>(actor);
    MS_EXCEPTION_IF_NULL(memory_aware_actor);
    auto alloc_actor_ptr = std::make_shared<MemoryAllocActor>(
      alloc_actor_name, memory_aware_actor->memory_manager_aid(), somas_info, device_contexts[0]);
    (void)actor_set->memory_actors_.emplace_back(alloc_actor_ptr);
    alloc_actor = alloc_actor_ptr.get();
    InsertActor(alloc_actor);

    // Link: insert_position_actor-->alloc_actor.
    SchedulerHelper::AddControlArrow(actor->memory_alloc_insert_position(), alloc_actor);
  }

  // Link: alloc_actor-->actor.
  SchedulerHelper::AddControlArrow(alloc_actor, actor);
}

void MemoryActorInsert::InsertMemoryFreeActor(ActorSet *const actor_set, AbstractActor *const actor,
                                              SomasInfo *somas_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(somas_info);

  std::string free_actor_name = "kernel_graph_" + std::to_string(somas_info->graph_id_) + kMemoryFreeActorNameSuffix;
  auto free_actor = FetchActor(free_actor_name);
  if (free_actor == nullptr) {
    // Build the memory free actor.
    auto &device_contexts = actor->device_contexts();
    MS_EXCEPTION_IF_CHECK_FAIL((!device_contexts.empty()), "The device context doesn't exist.");
    auto memory_aware_actor = dynamic_cast<MemoryAwareActor *>(actor);
    MS_EXCEPTION_IF_NULL(memory_aware_actor);
    auto free_actor_ptr = std::make_shared<MemoryFreeActor>(free_actor_name, memory_aware_actor->memory_manager_aid(),
                                                            somas_info, device_contexts[0]);
    (void)actor_set->memory_actors_.emplace_back(free_actor_ptr);
    free_actor = free_actor_ptr.get();
    InsertActor(free_actor);

    // Link: free_actor-->insert_position_actor.
    SchedulerHelper::AddControlArrow(free_actor, actor->memory_free_insert_position());
  }

  // Link: actor-->free_actor.
  SchedulerHelper::AddControlArrow(actor, free_actor);
}
}  // namespace runtime
}  // namespace mindspore
