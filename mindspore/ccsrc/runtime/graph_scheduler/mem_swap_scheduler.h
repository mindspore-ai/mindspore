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
#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_SWAP_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_SWAP_SCHEDULER_H_

#include <vector>
#include <map>
#include <utility>
#include <memory>

#include "utils/ms_utils.h"
#include "runtime/graph_scheduler/actor/memory/memory_swap_actor.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/graph_scheduler/actor/actor_set.h"

namespace mindspore {
namespace runtime {
class MemSwapScheduler {
 public:
  MemSwapScheduler() = default;
  ~MemSwapScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(MemSwapScheduler);

  std::vector<std::vector<MemSwapActorPtr>> Build(const GraphCompilerInfo &graph_compiler_info,
                                                  const AID *recorder_aid);
  void Link(const GraphCompilerInfo &graph_compiler_info, ActorSet *actor_set) const;

 private:
  void GetRealParameters(const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                         HashMap<AnfNodePtr, size_t> *real_parameters) const;

  void BuildSwapActorForGraph(const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                              const DeviceContext *device_context, std::vector<MemSwapActorPtr> *actors);
  AbstractActor *GetActorForLink(size_t id, const std::shared_ptr<device::SwapStrategy> &strategy,
                                 const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                                 ActorSet *actor_set) const;

 private:
  const AID *recorder_aid_;
  // KernelGraph id - SwapStrategy
  HashMap<size_t, std::shared_ptr<device::SwapStrategy>> graph_strategy_map_;
  // SwapAction id - MemSwapActor
  HashMap<size_t, MemSwapActorPtr> action_actor_map_;
  // MemSwapActorPtr - output index of EntranceActor for data dependency
  HashMap<MemSwapActorPtr, std::vector<size_t>> data_dependency_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_SWAP_SCHEDULER_H_
