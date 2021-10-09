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
#include "runtime/framework/actor/abstract_actor.h"

namespace mindspore {
namespace runtime {
// Stack actor is used to record those device actors that need additional storage in recursive scenes.
class StackActor : public MemoryAwareActor {
 public:
  StackActor(const std::string &name, const std::vector<KernelWithIndex> &parameters)
      : AbstractActor(name, KernelTransformType::kStackActor, nullptr), formal_parameters_(parameters) {
    device_contexts_.resize(parameters.size());
  }
  ~StackActor() override = default;

  void Init() override;

  // The stack actor run when receive the real parameter nodes.
  void CollectRealParameter(const AnfNodePtr &node, size_t index, size_t position,
                            OpContext<DeviceTensor> *const context);

 private:
  friend class GraphScheduler;

  // Formal parameters record the input front-end node, these nodes may be parameter, kernel, call node.
  std::vector<KernelWithIndex> formal_parameters_;

  // The backend parameter is used to save the backend node corresponding to the device tensor in the stack.
  // When these device tensors are used as output, they need to be placed in the node of the result arrow,
  // so these nodes need to be saved.
  std::vector<KernelWithIndex> backend_parameters_;

  // Input data.
  std::unordered_map<uuids::uuid *, std::unordered_map<size_t, KernelWithIndex>> input_nodes_;

  // The input data records that the stack actor is copied from the input nodes and needs to be stored in the
  // device tensor in the stack. This part of the device tensor does not belong to any node, and it will be
  // cleaned up directly after the stack is popped.
  std::unordered_map<uuids::uuid *, std::unordered_map<size_t, std::stack<DeviceTensor *>>> input_data_;
};

using StackActorPtr = std::shared_ptr<StackActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_STACK_ACTOR_H_
