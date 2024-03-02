/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_INLINE_CONTROL_FLOW_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_INLINE_CONTROL_FLOW_SCHEDULER_H_

#include "runtime/graph_scheduler/actor/actor_set.h"

namespace mindspore {
namespace runtime {
class InlineControlFlowScheduler {
 public:
  InlineControlFlowScheduler() = default;
  ~InlineControlFlowScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(InlineControlFlowScheduler);

  // Transform the condition switch and condition gather actor.
  void Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_INLINE_CONTROL_FLOW_SCHEDULER_H_
