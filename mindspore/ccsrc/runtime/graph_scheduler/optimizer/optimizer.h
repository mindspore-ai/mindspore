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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_OPTIMIZER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_OPTIMIZER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "runtime/graph_scheduler/actor/actor_set.h"

namespace mindspore {
namespace runtime {
class ActorPass {
 public:
  explicit ActorPass(const std::string &name, bool need_run_single_actor = true)
      : name_(name), need_run_single_actor_(need_run_single_actor) {}
  virtual ~ActorPass() = default;
  DISABLE_COPY_AND_ASSIGN(ActorPass);

  // The pass running flow: MatchPattern-->Process.
  void Run(const ActorSetPtr &actor_set);

  const std::string &name() const { return name_; }

 protected:
  virtual bool MatchPattern(const AbstractActor *actor) const { return true; }
  virtual void Process(ActorSet *const actor_set, AbstractActor *const actor) = 0;

 private:
  std::string name_;

  // Indicate the pass processing each actor or processing the whole actor set.
  bool need_run_single_actor_;
};
using ActorPassPtr = std::shared_ptr<ActorPass>;

class ActorSetOptimizer {
 public:
  ActorSetOptimizer() = default;
  ~ActorSetOptimizer() = default;
  DISABLE_COPY_AND_ASSIGN(ActorSetOptimizer);

  // Add the pass to the passes list.
  void AddPass(const ActorPassPtr &pass);

  // Foreach passes list to optimize the actor set.
  void Optimize(const ActorSetPtr &actor_set);

 private:
  std::string GetPassFullName(const ActorSetPtr &actor_set, const std::string &pass_name, size_t pass_id) const;
  void DumpPassActorSet(const ActorSetPtr &actor_set, const std::string &pass_full_name) const;

  std::vector<ActorPassPtr> passes_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_OPTIMIZER_OPTIMIZER_H_
