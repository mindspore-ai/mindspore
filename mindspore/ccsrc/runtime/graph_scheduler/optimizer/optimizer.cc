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

#include "runtime/graph_scheduler/optimizer/optimizer.h"
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "include/common/debug/common.h"
#include "utils/profile.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
namespace runtime {
void ActorPass::Run(const ActorSetPtr &actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  if (!need_run_single_actor_) {
    Process(actor_set.get(), nullptr);
  } else {
    auto actors = SchedulerHelper::CollectActors(actor_set.get());
    for (const auto &actor : actors) {
      if (!MatchPattern(actor.get())) {
        continue;
      }
      Process(actor_set.get(), actor.get());
    }
  }
}

void ActorSetOptimizer::AddPass(const ActorPassPtr &pass) {
  MS_EXCEPTION_IF_NULL(pass);
  (void)passes_.emplace_back(pass);
}

void ActorSetOptimizer::Optimize(const ActorSetPtr &actor_set) {
  size_t pass_id = 1;
  for (const auto &pass : passes_) {
    MS_EXCEPTION_IF_NULL(pass);
    double start_time = GetTime();
    pass->Run(actor_set);
    double end_time = GetTime();
    const size_t kSecondsToMicroseconds = 1000000;
    auto pass_full_name = GetPassFullName(actor_set, pass->name(), pass_id);
    MS_LOG(INFO) << "Run pass " + pass_full_name + " in " << (end_time - start_time) * kSecondsToMicroseconds << " us";
    DumpPassActorSet(actor_set, pass_full_name);
    ++pass_id;
  }
}

std::string ActorSetOptimizer::GetPassFullName(const ActorSetPtr &actor_set, const std::string &pass_name,
                                               size_t pass_id) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  return std::to_string(pass_id) + "_actor_set_" + actor_set->name_ + "_" + pass_name;
}

void ActorSetOptimizer::DumpPassActorSet(const ActorSetPtr &actor_set, const std::string &pass_full_name) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (!context->CanDump(kIntroductory)) {
    return;
  }

  // Get the saved actor set name.
  std::string save_name = "actor_set/" + pass_full_name;
  std::string path_name = GetSaveGraphsPathName(save_name + ".ir");
  auto realpath_name = Common::CreatePrefixPath(path_name);
  if (!realpath_name.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path: " << path_name;
    return;
  }

  ChangeFileMode(realpath_name.value(), S_IWUSR);
  std::ofstream ofs(realpath_name.value());
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << realpath_name.value() << "] failed!";
    return;
  }
  SchedulerHelper::DumpActorSet(actor_set.get(), ofs);
  ChangeFileMode(realpath_name.value(), S_IRUSR);
}
}  // namespace runtime
}  // namespace mindspore
