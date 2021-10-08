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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_ENTRANCE_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_ENTRANCE_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stack>
#include <queue>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/abstract_actor.h"

namespace mindspore {
namespace runtime {
// Entrance actor is used in the control flow to receive a set of result arrow and a branch id and then send
// the data to the corresponding actor. It is the entry point for subgraph execution.
class EntranceActor : public AbstractActor {
 public:
  EntranceActor(const std::string &name, const std::vector<AnfNodePtr> &parameters)
      : AbstractActor(name, KernelTransformType::kEntranceActor, nullptr), formal_parameters_(parameters) {
    device_contexts_.resize(parameters.size());
  }
  ~EntranceActor() override = default;

  void Init() override;

  // The entrance actor run when receive the real parameter nodes and branch id.
  void CollectRealParametersAndBranchId(const std::vector<KernelWithIndex> &real_parameters, int branch_id,
                                        OpContext<DeviceTensor> *const context);

 protected:
  void Run(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  // Formal parameters of actor, which is the front node.
  std::vector<KernelWithIndex> formal_parameters_;

  // Input data.
  std::unordered_map<uuids::uuid *, std::queue<std::vector<KernelWithIndex>>> input_nodes_;
  std::unordered_map<uuids::uuid *, std::queue<int>> input_branch_ids_;

  std::vector<AID> output_branch_id_arrows_;
  // The output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<OpData<DeviceTensor> *> output_data_;
  bool is_actor_ready_{true};
};

using EntranceActorPtr = std::shared_ptr<EntranceActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_ENTRANCE_ACTOR_H_
