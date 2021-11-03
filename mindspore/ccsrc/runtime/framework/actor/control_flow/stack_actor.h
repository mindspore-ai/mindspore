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
#include <unordered_map>
#include <stack>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/control_flow/control_actor.h"

namespace mindspore {
namespace runtime {
// Stack actor is used to record those device actors that need additional storage in recursive scenes.
// The execution steps of the stack actor:
// 1. Accept a copy of all direct parameters and push them to the stack
// 2. Notify gather actor can be executed
// 3. Receive the output of exit actor
// 4. send output.
class StackActor : public ControlActor {
 public:
  StackActor(const std::string &name, const std::vector<KernelWithIndex> &parameters);
  ~StackActor() override = default;

 protected:
  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context);
  void FetchInput(OpContext<DeviceTensor> *const context);
  bool CheckRunningCondition(const OpContext<DeviceTensor> *context) const;
  void EraseInput(const OpContext<DeviceTensor> *const context);

 private:
  friend class ControlNodeScheduler;

  // The backend parameter is used to save the backend node corresponding to the device tensor in the stack.
  // When these device tensors are used as output, they need to be placed in the node of the result arrow,
  // so these nodes need to be saved.
  std::vector<KernelWithIndex> backend_parameters_;
};

using StackActorPtr = std::shared_ptr<StackActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_STACK_ACTOR_H_
