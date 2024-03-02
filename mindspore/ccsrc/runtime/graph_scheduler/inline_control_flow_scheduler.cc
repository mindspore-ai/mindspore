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

#include "runtime/graph_scheduler/inline_control_flow_scheduler.h"
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "ops/framework_ops.h"

namespace mindspore {
namespace runtime {
void InlineControlFlowScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) != kOptimizeO0) {
    for (const auto &graph : graph_compiler_info.graphs_) {
      MS_EXCEPTION_IF_NULL(graph);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
