/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_STACK_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_STACK_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <stack>
#include <set>
#include <algorithm>
#include "utils/hash_map.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/control_flow/control_actor.h"

namespace mindspore {
namespace runtime {
// Stack actor is used to record those device actors that need additional storage in recursive scenes.
// The execution steps of the stack actor:
// 1. Accept a copy of all direct parameters and push them to the stack
// 2. Notify gather actor can be executed
// 3. Receive the output of exit actor
// 4. Send output.
class StackActor : public ControlActor {
 public:
  StackActor(const std::string &name, const AID &memory_manager_aid, const std::vector<KernelWithIndex> &parameters);
  ~StackActor() override = default;

  // The input data and partial of the stack actor needs to be pushed into the stack according to the input index,
  // so it is implemented separately.
  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) override;
  void RunOpPartial(const OpPartialPtr &partial, size_t position, OpContext<DeviceTensor> *const context) override;
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;

  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;

  size_t input_stack_data_num() const { return input_stack_data_num_; }
  size_t input_stack_partials_num() const { return input_stack_partials_num_; }
  size_t input_stack_controls_num() const { return input_stack_controls_num_; }

 protected:
  void Init() override;
  void FetchInput(OpContext<DeviceTensor> *const context) override;
  bool CheckRunningCondition(const OpContext<DeviceTensor> *context) const override;
  void EraseInput(const OpContext<DeviceTensor> *const context) override;

 private:
  friend class ControlNodeScheduler;

  // Check running condition functions.
  bool CheckStackDataRunningCondition(const OpContext<DeviceTensor> *context) const;
  bool CheckStackPartialRunningCondition(const OpContext<DeviceTensor> *context) const;
  bool CheckStackControlRunningCondition(const OpContext<DeviceTensor> *context) const;

  // The input data and partials records that the stack actor is copied from the input nodes and needs to be
  // stored in the device tensor in the stack.
  mindspore::HashMap<int, mindspore::HashMap<size_t, std::stack<DeviceTensor *>>> input_stack_data_;
  mindspore::HashMap<int, mindspore::HashMap<size_t, std::stack<OpPartialPtr>>> input_stack_partials_;
  // When the input has side effects, some control arrows need to be pushed to the stack, which needs to be
  // recorded according to the from aids, but if the input node is a call node, the input control arrows may
  // come from different exit actors, so the relationship between from actor and index needs to be recorded
  // during the schedule, and the number of control arrows is recorded according to the index at runtime.
  mindspore::HashMap<AID, size_t> control_aid_to_indexs_;
  mindspore::HashMap<int, mindspore::HashMap<size_t, size_t>> input_stack_controls_;
  std::set<AID> stack_control_aids_;
  // Input parameter num represents the number of actor's input come from funcgraph itself, these inputs will
  // be ranked at the front of input.
  size_t input_stack_data_num_{0};
  size_t input_stack_partials_num_{0};
  size_t input_stack_controls_num_{0};
};

using StackActorPtr = std::shared_ptr<StackActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_STACK_ACTOR_H_
