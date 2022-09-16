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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_MEMORY_ACTOR_INSERT_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_MEMORY_ACTOR_INSERT_H_

#include <memory>
#include "runtime/graph_scheduler/optimizer/optimizer.h"

namespace mindspore {
namespace runtime {
// Insert the memory alloc and free actors at the boundary of the graph for integration of dynamic and static memory.
class MemoryActorInsert : public ActorPass {
 public:
  MemoryActorInsert() : ActorPass("memory_actor_insert") {}
  ~MemoryActorInsert() override = default;

 protected:
  bool MatchPattern(const AbstractActor *actor) const override;
  void Process(ActorSet *const actor_set, AbstractActor *const actor) override;

 private:
  void InsertMemoryAllocActor(ActorSet *const actor_set, AbstractActor *const actor, SomasInfo *somas_info) const;
  void InsertMemoryFreeActor(ActorSet *const actor_set, AbstractActor *const actor, SomasInfo *somas_info) const;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_MEMORY_ACTOR_INSERT_H_
