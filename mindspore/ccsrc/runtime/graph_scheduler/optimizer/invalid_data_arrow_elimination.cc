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

#include "runtime/graph_scheduler/optimizer/invalid_data_arrow_elimination.h"
#include <vector>
#include "runtime/graph_scheduler/scheduler_helper.h"

namespace mindspore {
namespace runtime {
namespace {
// Judge the from_data_arrow whether is in the to_actor.
bool IsDataArrowInActor(const DataArrowPtr &from_data_arrow, const AbstractActor *to_actor) {
  MS_EXCEPTION_IF_NULL(from_data_arrow);
  MS_EXCEPTION_IF_NULL(to_actor);

  if (from_data_arrow->to_op_id_ != to_actor->GetAID()) {
    return false;
  }
  return std::any_of(to_actor->output_data_arrows().begin(), to_actor->output_data_arrows().end(),
                     [&from_data_arrow](const DataArrowPtr &to_data_arrow) {
                       return from_data_arrow->to_input_index_ == to_data_arrow->from_output_index_;
                     });
}
}  // namespace

bool InvalidDataArrowElimination::MatchPattern(const AbstractActor *actor) const {
  MS_EXCEPTION_IF_NULL(actor);
  if (actor->type() == KernelTransformType::kExitActor) {
    auto exit_actor = dynamic_cast<ExitActor *>(const_cast<AbstractActor *>(actor));
    MS_EXCEPTION_IF_NULL(exit_actor);
    // Only handle the exit actor from kernel graph.
    if (exit_actor->node() == nullptr) {
      return true;
    }
  }
  return false;
}

void InvalidDataArrowElimination::Process(ActorSet *const actor_set, AbstractActor *const exit_actor) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(exit_actor);

  // The input_data_arrow_aids_ of exit actor will be changed in the ConvertDataArrowToControlArrow, so need copy.
  auto input_data_arrow_aids = exit_actor->input_data_arrow_aids();
  for (auto &input_data_arrow_aid : input_data_arrow_aids) {
    auto input_actor = FetchActor(input_data_arrow_aid.first.Name());
    MS_EXCEPTION_IF_NULL(input_actor);
    MS_EXCEPTION_IF_CHECK_FAIL((input_actor != nullptr), (input_data_arrow_aid.first.Name() + " is nullptr."));
    // Only handle the kernel actor to kernel graph exit actor.
    if (input_actor->type() != KernelTransformType::kKernelActor) {
      continue;
    }

    std::vector<DataArrowPtr> no_used_arrows;
    std::vector<size_t> no_used_arrow_indices;
    // Get all the no used arrows.
    auto &output_data_arrows = input_actor->output_data_arrows();
    for (size_t i = 0; i < output_data_arrows.size(); ++i) {
      auto &output_data_arrow = output_data_arrows[i];
      MS_EXCEPTION_IF_NULL(output_data_arrow);
      // Skip the valid data arrow.
      if ((output_data_arrow->to_op_id_ != exit_actor->GetAID()) ||
          (IsDataArrowInActor(output_data_arrow, exit_actor))) {
        continue;
      }
      (void)no_used_arrows.emplace_back(output_data_arrow);
      (void)no_used_arrow_indices.emplace_back(i);
    }

    // Convert the no used data arrow to control arrow backward to avoid the vector index error.
    for (size_t arrow_index = no_used_arrows.size(); arrow_index > 0; --arrow_index) {
      SchedulerHelper::ConvertDataArrowToControlArrow(input_actor, exit_actor, no_used_arrows[arrow_index - 1],
                                                      no_used_arrow_indices[arrow_index - 1]);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
