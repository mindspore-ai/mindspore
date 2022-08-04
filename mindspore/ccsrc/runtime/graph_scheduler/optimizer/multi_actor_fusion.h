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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_MULTI_ACTOR_FUSION_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_MULTI_ACTOR_FUSION_H_

#include <memory>
#include <utility>
#include <string>
#include <set>
#include "runtime/graph_scheduler/optimizer/optimizer.h"

namespace mindspore {
namespace runtime {
// Fuse the actors which have the execution dependency to a big actor. These actors can't execute concurrently.
class MultiActorFusion : public ActorPass {
 public:
  MultiActorFusion() : ActorPass("multi_actor_fusion", false) {}
  ~MultiActorFusion() override = default;

 protected:
  void Process(ActorSet *const actor_set, AbstractActor *const actor) override;

 private:
  bool AnalyzeDependency(const ActorSet *actor_set) const;
  bool AddDependency(std::pair<AbstractActor *, bool> *const actor_info,
                     mindspore::HashMap<std::string, std::pair<AbstractActor *, bool>> *const actor_infos) const;

  void FuseMultiActors(ActorSet *const actor_set) const;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_MULTI_ACTOR_FUSION_H_
