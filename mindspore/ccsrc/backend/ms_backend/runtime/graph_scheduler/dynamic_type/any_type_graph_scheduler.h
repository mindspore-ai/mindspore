/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ANY_TYPE_GRAPH_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ANY_TYPE_GRAPH_SCHEDULER_H_

#include <map>
#include <utility>
#include <vector>

#include "utils/ms_utils.h"
#include "backend/ms_backend/runtime/actors/base/actor_set.h"

namespace mindspore {
namespace runtime {
class AnyTypeGraphScheduler {
 public:
  AnyTypeGraphScheduler() = default;
  ~AnyTypeGraphScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(AnyTypeGraphScheduler);
  std::vector<AnyTypeKernelActorPtr> Build(const GraphCompilerInfo &graph_compiler_info, const AID &memory_manager_aid,
                                           const AID *debug_id);
  void Optimize(const ActorSetPtr &actor_set,
                const std::map<KernelWithIndex, std::pair<AbstractActor *, KernelWithIndex>,
                               session::KernelWithIndexCmp> &graph_output_to_actor) const;
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ANY_TYPE_GRAPH_SCHEDULER_H_
