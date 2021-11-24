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
#include <stack>
#include <queue>
#include <set>
#include "utils/hash_map.h"
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/control_flow/control_actor.h"

namespace mindspore {
namespace runtime {
// Entrance actor is used in the control flow to receive a set of result arrow and a branch id and then send
// the data to the corresponding actor. It is the entry point for subgraph execution.
class EntranceActor : public ControlActor {
 public:
  EntranceActor(const std::string &name, const std::vector<KernelWithIndex> &parameters,
                const std::set<KernelWithIndex> &call_nodes, const AnfNodePtr &node)
      : ControlActor(name, KernelTransformType::kEntranceActor, parameters, node), call_nodes_(call_nodes) {
    device_contexts_.resize(parameters.size());
    input_device_tensors_.resize(parameters.size());
  }
  ~EntranceActor() override = default;

  void RunOpDataWithBranchID(std::vector<DeviceTensor *> input_data, int branch_id,
                             OpContext<DeviceTensor> *const context);

 protected:
  void Run(OpContext<DeviceTensor> *const context) override;
  void FetchInput(OpContext<DeviceTensor> *const context) override;
  bool CheckRunningCondition(const OpContext<DeviceTensor> *context) const override;
  void EraseInput(const OpContext<DeviceTensor> *const context) override;

 private:
  friend class ControlNodeScheduler;

  // Check if actor is enable. During operation, entrance actor can be enabled only when receives all control arrows.
  bool CheckActorStatus(const OpContext<DeviceTensor> *const context) const;

  // Is actor ready indicates whether the entrance actor can be executed. In the control flow, the subgraph is an
  // atomic operation, and execution can only continue after the output of the corresponding exit actor is completed.
  // At this time, the exit actor will notify the entrance actor to change the ready to true.
  bool is_actor_ready_{true};

  // Input data with branch id.
  mindspore::HashMap<int, std::queue<OpDataWithBranchID>> input_op_data_with_branch_id_;

  // Call nodes are used to record the caller of the subgraph, and are used to connect the data arrow
  // and branch id arrow in the link process.
  std::set<KernelWithIndex> call_nodes_;
};

using EntranceActorPtr = std::shared_ptr<EntranceActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_ENTRANCE_ACTOR_H_
