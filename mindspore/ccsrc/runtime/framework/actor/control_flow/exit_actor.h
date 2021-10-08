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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_EXIT_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_EXIT_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stack>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/abstract_actor.h"

namespace mindspore {
namespace runtime {
// The exit actor is used to receive a set of result arrow and a branch id in the control flow, and then send the
// node in the result to the corresponding actor. It is the exit of the end of subgraph execution.
class ExitActor : public AbstractActor {
 public:
  ExitActor(const std::string &name, const std::vector<AnfNodePtr> &parameters)
      : AbstractActor(name, KernelTransformType::kExitActor, nullptr), formal_parameters_(parameters) {}
  ~ExitActor() override = default;

  // The exit actor run when receive the anfnode.
  void CollectRealParameter(const AnfNodePtr &output_node, size_t output_index, size_t output_position,
                            OpContext<DeviceTensor> *const context);
  // The exit actor run when receive the input branch id.
  void CollectBranchId(int branch_id, OpContext<DeviceTensor> *const context);

 private:
  friend class GraphScheduler;

  // Formal parameters of actor, which is the front node.
  std::vector<KernelWithIndex> formal_parameters_;

  // Input data.
  std::unordered_map<uuids::uuid *, std::unordered_map<size_t, KernelWithIndex>> input_nodes_;
  // Branch ids is used to record the id corresponding to the output branch.
  // In control flow, sub funcgraph may be called in multiple places, and the output must be return to different
  // places. Therefore, the output of each subgraph will be connected to a exit actor, and the caller will send
  // its branch id to the entrance actor of the subgraph. Then branch id will be sent by the entrance actor to
  // the exit actor connected to the output.
  // In a recursive scenario, the exit will sequentially receive the branch ids sent by the caller, and the exit
  // actor needs to store the branch ids in the stack, and pop up in turn when returning.
  std::unordered_map<uuids::uuid *, std::stack<int>> input_branch_ids_;

  // Output arrow.
  std::unordered_map<int, std::vector<DataArrowPtr>> output_branch_result_arrows_;
};

using ExitActorPtr = std::shared_ptr<ExitActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_EXIT_ACTOR_H_
