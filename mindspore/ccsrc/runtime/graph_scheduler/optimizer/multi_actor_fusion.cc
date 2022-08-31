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
#include <queue>
#include "runtime/graph_scheduler/scheduler_helper.h"

namespace mindspore {
namespace runtime {
namespace {
bool SupportFusion(const AbstractActorPtr &actor) {
  MS_EXCEPTION_IF_NULL(actor);
  if ((actor->type() == KernelTransformType::kDeviceDataSourceActor) ||
      (actor->type() == KernelTransformType::kHostDataSourceActor) ||
      (actor->type() == KernelTransformType::kKernelActor) ||
      (actor->type() == KernelTransformType::kSuperKernelActor) || (actor->type() == KernelTransformType::kCopyActor)) {
    return true;
  }
  return false;
}
}  // namespace

// The max actors num in fusion actor.
constexpr size_t kActorFusionMaxNum = 1000;

void MultiActorFusion::Process(ActorSet *const actor_set, AbstractActor *const) {
  MS_EXCEPTION_IF_NULL(actor_set);
  if (!actor_set->custom_actors_.empty()) {
    return;
  }

  // Build all the fusion actors.
  FuseMultiActors(actor_set);

  // Link fusion actor.
  for (auto &fusion_actor : actor_set->fusion_actors_) {
    SchedulerHelper::AddArrowForFusionActor(fusion_actor.get());
  }
}

bool MultiActorFusion::AnalyzeDependency(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto need_processed_actors = SchedulerHelper::CollectActors(actor_set);
  // The second of pair indicates whether the actor finishes adding the dependency.
  mindspore::HashMap<std::string, std::pair<AbstractActor *, bool>> actor_infos;
  for (auto &actor : need_processed_actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (!SupportFusion(actor)) {
      actor_infos[actor->GetAID().Name()] = std::make_pair(actor.get(), true);
    } else {
      actor_infos[actor->GetAID().Name()] = std::make_pair(actor.get(), false);
    }
  }

  std::vector<AbstractActorPtr> unprocessed_actors;
  while (!need_processed_actors.empty()) {
    MS_LOG(INFO) << actor_set->name_ << " analyze dependency and process actors num: " << need_processed_actors.size();
    for (auto &actor : need_processed_actors) {
      MS_EXCEPTION_IF_NULL(actor);
      auto &current_actor_info = actor_infos[actor->GetAID().Name()];
      // Maybe finish adding the dependency in the function AddDependency.
      if (current_actor_info.second) {
        continue;
      }

      // Collect the input actor infos from the input data and input control.
      std::set<std::pair<AbstractActor *, bool>> input_actor_infos;
      for (auto &input_data_arrow_aid : actor->input_data_arrow_aids()) {
        (void)input_actor_infos.insert(actor_infos[input_data_arrow_aid.first.Name()]);
      }
      for (auto &input_control_arrow_aid : actor->input_control_arrow_aids()) {
        (void)input_actor_infos.insert(actor_infos[input_control_arrow_aid.first.Name()]);
      }

      // Add the dependency from the input actor info.
      current_actor_info.second = true;
      for (auto &input_actor_info : input_actor_infos) {
        if (!AddDependency(const_cast<std::pair<AbstractActor *, bool> *>(&input_actor_info), &actor_infos)) {
          (void)unprocessed_actors.emplace_back(actor);
          current_actor_info.second = false;
          break;
        }
        SchedulerHelper::AddDependency(actor.get(), input_actor_info.first);
      }
    }

    // This iteration doesn't process any actor and need stop.
    if (need_processed_actors.size() == unprocessed_actors.size()) {
      return false;
    }
    // Updata the actors which need be processed in the next iteration.
    need_processed_actors.assign(unprocessed_actors.begin(), unprocessed_actors.end());
    unprocessed_actors.clear();
  }

  return true;
}

bool MultiActorFusion::AddDependency(
  std::pair<AbstractActor *, bool> *const actor_info,
  mindspore::HashMap<std::string, std::pair<AbstractActor *, bool>> *const actor_infos) const {
  MS_EXCEPTION_IF_NULL(actor_info);
  MS_EXCEPTION_IF_NULL(actor_infos);
  if (actor_info->second) {
    return true;
  }

  // Collect the input actor infos from the input data and input control.
  MS_EXCEPTION_IF_NULL(actor_info->first);
  std::set<std::pair<AbstractActor *, bool>> input_actor_infos;
  for (auto &input_data_arrow_aid : actor_info->first->input_data_arrow_aids()) {
    (void)input_actor_infos.insert(actor_infos->at(input_data_arrow_aid.first.Name()));
  }
  for (auto &input_control_arrow_aid : actor_info->first->input_control_arrow_aids()) {
    (void)input_actor_infos.insert(actor_infos->at(input_control_arrow_aid.first.Name()));
  }

  // Add the dependency from the input actor info.
  for (auto &input_actor_info : input_actor_infos) {
    if (!input_actor_info.second) {
      return false;
    }
    SchedulerHelper::AddDependency(actor_info->first, input_actor_info.first);
  }

  actor_info->second = true;
  return true;
}

namespace {
bool GetDependentActors(std::vector<AbstractActorPtr> *const output_actors, const AbstractActorPtr &actor,
                        mindspore::HashMap<std::string, AbstractActorPtr> *const need_processed_actors) {
  MS_EXCEPTION_IF_NULL(output_actors);
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(need_processed_actors);

  bool is_need_processed = SupportFusion(actor) ? true : false;
  std::vector<std::string> output_actor_names;
  // Get all the output actors by output data.
  for (auto &output_data_arrow : actor->output_data_arrows()) {
    MS_EXCEPTION_IF_NULL(output_data_arrow);
    (void)output_actor_names.emplace_back(output_data_arrow->to_op_id_.Name());
  }
  // Get all the output actors by output control.
  for (auto &output_control_arrow : actor->output_control_arrows()) {
    MS_EXCEPTION_IF_NULL(output_control_arrow);
    (void)output_actor_names.emplace_back(output_control_arrow->to_op_id_.Name());
  }
  // Get all the output actors by output partial of control actor.
  if (IsControlFlowActor(actor->type())) {
    auto control_actor = dynamic_cast<ControlActor *>(actor.get());
    MS_EXCEPTION_IF_NULL(control_actor);
    for (auto &output_partial_arrow : control_actor->output_partial_arrows()) {
      MS_EXCEPTION_IF_NULL(output_partial_arrow);
      (void)output_actor_names.emplace_back(output_partial_arrow->to_op_id_.Name());
    }
  }
  // Get all the output actors by output data with branch id of gather actor.
  if (actor->type() == KernelTransformType::kGatherActor) {
    auto gather_actor = dynamic_cast<GatherActor *>(actor.get());
    MS_EXCEPTION_IF_NULL(gather_actor);
    for (auto &output_data_with_branch_id_arrow : gather_actor->output_data_with_branch_id_arrows()) {
      for (auto &output_aid : output_data_with_branch_id_arrow.second) {
        (void)output_actor_names.emplace_back(output_aid.Name());
      }
    }
  }

  for (auto &output_actor_name : output_actor_names) {
    // Skip the repeated output.
    if (std::find_if(output_actors->begin(), output_actors->end(), [&output_actor_name](auto &output_actor) {
          return output_actor_name == output_actor->GetAID().Name();
        }) != output_actors->end()) {
      continue;
    }
    auto iter = need_processed_actors->find(output_actor_name);
    if (iter != need_processed_actors->end()) {
      (void)output_actors->emplace_back(iter->second);
      if (!SupportFusion(iter->second)) {
        is_need_processed = false;
      }
      (void)need_processed_actors->erase(iter);
    } else {
      is_need_processed = false;
    }
  }

  return (is_need_processed && (output_actors->size() == 1));
}

std::vector<FusionActorPtr> BuildFusionActorBySeed(
  const AbstractActorPtr &seed_actor, std::queue<AbstractActorPtr> *const origin_seed_actors,
  mindspore::HashMap<std::string, AbstractActorPtr> *const need_processed_actors) {
  MS_EXCEPTION_IF_NULL(seed_actor);
  MS_EXCEPTION_IF_NULL(origin_seed_actors);
  MS_EXCEPTION_IF_NULL(need_processed_actors);

  // Get the actors need be fused.
  std::vector<AbstractActorPtr> need_fused_actors;
  std::queue<AbstractActorPtr> current_seed_actors;
  current_seed_actors.push(seed_actor);
  while (!current_seed_actors.empty()) {
    auto current_seed_actor = current_seed_actors.front();
    current_seed_actors.pop();
    if (SupportFusion(current_seed_actor)) {
      (void)need_fused_actors.emplace_back(current_seed_actor);
    }

    // Get the outputs of seed actor. If they have dependencies, continue processing. Otherwise, add the output to
    // origin_seed_actors.
    std::vector<AbstractActorPtr> output_actors;
    bool has_dependency = GetDependentActors(&output_actors, current_seed_actor, need_processed_actors);
    for (auto &output_actor : output_actors) {
      if (has_dependency) {
        current_seed_actors.push(output_actor);
      } else {
        origin_seed_actors->push(output_actor);
      }
    }
    output_actors.clear();
  }

  // Build the fusion actors.
  std::vector<FusionActorPtr> output_fused_actors;
  std::vector<AbstractActorPtr> sub_fused_actors;
  size_t i = 0;
  for (; i < (need_fused_actors.size() / kActorFusionMaxNum); ++i) {
    auto first =
      need_fused_actors.begin() + static_cast<std::vector<AbstractActorPtr>::difference_type>(kActorFusionMaxNum * i);
    auto end = need_fused_actors.begin() +
               static_cast<std::vector<AbstractActorPtr>::difference_type>(kActorFusionMaxNum * (i + 1));
    sub_fused_actors.assign(first, end);
    (void)output_fused_actors.emplace_back(SchedulerHelper::BuildFusionActor(sub_fused_actors));
    sub_fused_actors.clear();
  }
  sub_fused_actors.assign(
    need_fused_actors.begin() + static_cast<std::vector<AbstractActorPtr>::difference_type>(kActorFusionMaxNum * i),
    need_fused_actors.end());
  if (sub_fused_actors.size() > 1) {
    (void)output_fused_actors.emplace_back(SchedulerHelper::BuildFusionActor(sub_fused_actors));
  }

  return output_fused_actors;
}
}  // namespace

void MultiActorFusion::FuseMultiActors(ActorSet *const actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->data_prepare_actor_);

  auto actors = SchedulerHelper::CollectActors(actor_set);
  mindspore::HashMap<std::string, AbstractActorPtr> need_processed_actors;
  for (auto &actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    need_processed_actors[actor->GetAID().Name()] = actor;
  }

  // Get the initial seed actors from the outputs of data prepare actor.
  (void)need_processed_actors.erase(actor_set->data_prepare_actor_->GetAID().Name());
  std::queue<AbstractActorPtr> seed_actors;
  for (auto &output_control_arrow : actor_set->data_prepare_actor_->output_control_arrows()) {
    MS_EXCEPTION_IF_NULL(output_control_arrow);
    auto iter = need_processed_actors.find(output_control_arrow->to_op_id_.Name());
    if (iter != need_processed_actors.end()) {
      seed_actors.push(iter->second);
      (void)need_processed_actors.erase(iter);
    }
  }

  while (!seed_actors.empty()) {
    auto seed_actor = seed_actors.front();
    seed_actors.pop();
    MS_EXCEPTION_IF_NULL(seed_actor);
    auto fusion_actors = BuildFusionActorBySeed(seed_actor, &seed_actors, &need_processed_actors);
    for (auto &fusion_actor : fusion_actors) {
      (void)actor_set->fusion_actors_.emplace_back(fusion_actor);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
