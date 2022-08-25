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
const int32_t kInvalidInsertPosition = -1;

bool MemoryActorInsert::MatchPattern(const AbstractActor *actor) const {
  MS_EXCEPTION_IF_NULL(actor);
  if (actor->type() != KernelTransformType::kKernelActor) {
    return false;
  }
  return true;
}

void MemoryActorInsert::Process(ActorSet *const actor_set, AbstractActor *const actor) {
  MS_EXCEPTION_IF_NULL(actor);
  auto kernel_actor = dynamic_cast<KernelActor *>(actor);
  MS_EXCEPTION_IF_NULL(kernel_actor);
  auto somas_info = kernel_actor->somas_info();
  if (somas_info == nullptr) {
    return;
  }

  InsertMemoryAllocActor(actor_set, kernel_actor, somas_info);
  InsertMemoryFreeActor(actor_set, kernel_actor, somas_info);
}

void MemoryActorInsert::InsertMemoryAllocActor(ActorSet *const actor_set, KernelActor *const kernel_actor,
                                               SomasInfo *somas_info) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  auto &memory_alloc_insert_position = kernel_actor->memory_alloc_insert_position();
  if (memory_alloc_insert_position.first == kInvalidInsertPosition) {
    return;
  }

  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(somas_info);
  std::string alloc_actor_name = "kernel_graph_" + std::to_string(somas_info->graph_id_) + kMemoryAllocActorNameSuffix;
  auto alloc_actor = FetchActor(alloc_actor_name);
  if (alloc_actor == nullptr) {
    // Build the memory alloc actor.
    auto &device_contexts = kernel_actor->device_contexts();
    MS_EXCEPTION_IF_CHECK_FAIL((!device_contexts.empty()), "The device context doesn't exist.");
    auto alloc_actor_ptr = std::make_shared<MemoryAllocActor>(alloc_actor_name, kernel_actor->memory_manager_aid(),
                                                              somas_info, device_contexts[0]);
    (void)actor_set->memory_actors_.emplace_back(alloc_actor_ptr);
    alloc_actor = alloc_actor_ptr.get();
    InsertActor(alloc_actor);

    // Get the from actor of kernel actor that need insert the memory alloc actor.
    size_t insert_position = IntToSize(memory_alloc_insert_position.first);
    std::string from_actor_name = "";
    if (memory_alloc_insert_position.second) {
      auto &input_data_arrow_aids = kernel_actor->input_data_arrow_aids();
      MS_EXCEPTION_IF_CHECK_FAIL((input_data_arrow_aids.size() > insert_position),
                                 "The memory alloc actor insertion position out of range.");
      from_actor_name = input_data_arrow_aids[insert_position].first.Name();
    } else {
      auto &input_control_arrow_aids = kernel_actor->input_control_arrow_aids();
      MS_EXCEPTION_IF_CHECK_FAIL((input_control_arrow_aids.size() > insert_position),
                                 "The memory alloc actor insertion position out of range.");
      from_actor_name = input_control_arrow_aids[insert_position].first.Name();
    }
    auto from_actor = FetchActor(from_actor_name);
    MS_EXCEPTION_IF_NULL(from_actor);

    // Link: from_actor-->alloc_actor.
    SchedulerHelper::AddControlArrow(from_actor, alloc_actor);
  }

  // Link: alloc_actor-->kernel_actor.
  SchedulerHelper::AddControlArrow(alloc_actor, kernel_actor);
}

void MemoryActorInsert::InsertMemoryFreeActor(ActorSet *const actor_set, KernelActor *const kernel_actor,
                                              SomasInfo *somas_info) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  auto &memory_free_insert_position = kernel_actor->memory_free_insert_position();
  if (memory_free_insert_position.first == kInvalidInsertPosition) {
    return;
  }

  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(somas_info);
  std::string free_actor_name = "kernel_graph_" + std::to_string(somas_info->graph_id_) + kMemoryFreeActorNameSuffix;
  auto free_actor = FetchActor(free_actor_name);
  if (free_actor == nullptr) {
    // Build the memory free actor.
    auto &device_contexts = kernel_actor->device_contexts();
    MS_EXCEPTION_IF_CHECK_FAIL((!device_contexts.empty()), "The device context doesn't exist.");
    auto free_actor_ptr = std::make_shared<MemoryFreeActor>(free_actor_name, kernel_actor->memory_manager_aid(),
                                                            somas_info, device_contexts[0]);
    (void)actor_set->memory_actors_.emplace_back(free_actor_ptr);
    free_actor = free_actor_ptr.get();
    InsertActor(free_actor);

    // Get the to actor of kernel actor that need insert the memory free actor.
    size_t insert_position = IntToSize(memory_free_insert_position.first);
    std::string to_actor_name = "";
    if (memory_free_insert_position.second) {
      auto &out_data_arrows = kernel_actor->output_data_arrows();
      MS_EXCEPTION_IF_CHECK_FAIL((out_data_arrows.size() > insert_position),
                                 "The memory free actor insertion position out of range.");
      MS_EXCEPTION_IF_NULL(out_data_arrows[insert_position]);
      to_actor_name = out_data_arrows[insert_position]->to_op_id_.Name();
    } else {
      auto &output_control_arrows = kernel_actor->output_control_arrows();
      MS_EXCEPTION_IF_CHECK_FAIL((output_control_arrows.size() > insert_position),
                                 "The memory free actor insertion position out of range.");
      MS_EXCEPTION_IF_NULL(output_control_arrows[insert_position]);
      to_actor_name = output_control_arrows[insert_position]->to_op_id_.Name();
    }
    auto to_actor = FetchActor(to_actor_name);
    MS_EXCEPTION_IF_NULL(to_actor);

    // Link: free_actor-->to_actor.
    SchedulerHelper::AddControlArrow(free_actor, to_actor);
  }

  // Link: kernel_actor-->free_actor.
  SchedulerHelper::AddControlArrow(kernel_actor, free_actor);
}
}  // namespace runtime
}  // namespace mindspore
