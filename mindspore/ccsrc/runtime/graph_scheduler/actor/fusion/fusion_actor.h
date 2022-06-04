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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_FUSION_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_FUSION_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "utils/hash_map.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/abstract_actor.h"

namespace mindspore {
namespace runtime {
// The fusion actor is the actors set that have the execution dependency. These actors can't execute concurrently and
// fuse to the FusionActor.
class FusionActor : public AbstractActor {
 public:
  explicit FusionActor(const std::string &name) : AbstractActor(name, KernelTransformType::kFusionActor, nullptr) {}
  ~FusionActor() override = default;

  const mindspore::HashMap<std::string, AbstractActorPtr> &actors() const { return actors_; }

 protected:
  void Run(OpContext<DeviceTensor> *const context) override;

 private:
  friend class SchedulerHelper;

  // These actors are not spawned in the ActorMgr, so they do not participate in message interaction, but only internal
  // processing.
  mindspore::HashMap<std::string, AbstractActorPtr> actors_;

  // std::pair<actor, input_index> used to find the mapping between fusion actor inputs and real actors inputs.
  std::vector<std::pair<AbstractActor *, size_t>> real_input_data_;
  // Used to record the input controls info of the real actors.
  std::vector<AbstractActor *> real_input_controls_;
};

using FusionActorPtr = std::shared_ptr<FusionActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_FUSION_ACTOR_H_
