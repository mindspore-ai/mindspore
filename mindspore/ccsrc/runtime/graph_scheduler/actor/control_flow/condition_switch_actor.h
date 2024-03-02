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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_SWITCH_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_SWITCH_ACTOR_H_

#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/kernel_actor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;

// Condition switch actor is used to execute the branch according to the input condition in kernel graph.
class ConditionSwitchActor : public KernelActor {
 public:
  ConditionSwitchActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                       const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                       GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                       const std::set<size_t> &modifiable_ref_output_indexes,
                       const KernelTransformType &type = KernelTransformType::kConditionSwitchActor);
  ~ConditionSwitchActor() override = default;
};

using ConditionSwitchActorPtr = std::shared_ptr<ConditionSwitchActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_SWITCH_ACTOR_H_
