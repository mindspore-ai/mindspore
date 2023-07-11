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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_GATHER_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_GATHER_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "utils/hash_map.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/control_flow/control_actor.h"

namespace mindspore {
namespace runtime {

// Gather actor will be used in the control flow. When the subgraph is called, the real parameters need to be put
// together and sent to the subgraph.
class GatherActor : public ControlActor {
 public:
  GatherActor(const std::string &name, const AID &memory_manager_aid, const std::vector<KernelWithIndex> &parameters,
              const AnfNodePtr &node);
  ~GatherActor() override = default;

  const mindspore::HashMap<FuncGraph *, std::vector<AID>> &output_data_with_branch_id_arrows() const {
    return output_data_with_branch_id_arrows_;
  }
  const mindspore::HashMap<FuncGraph *, std::vector<std::pair<std::vector<size_t>, bool>>> &dynamic_len_index() const {
    return dynamic_len_index_;
  }

 protected:
  void SendOutput(OpContext<DeviceTensor> *const context) override;
  void IncreaseDynamicRefCounts(OpContext<DeviceTensor> *const context) override;

 private:
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;

  // Gather the input data and input partials to a new partial.
  void GatherInput(OpContext<DeviceTensor> *const context);

  void BuildOutput(OpRealParameterWithBranchID *const output, OpContext<DeviceTensor> *const context);

  // The input gathered by input data and input partials, which is created in GatherInput and destroyed in SendOutput.
  OpPartialPtr gather_input_;

  // There will be multiple output branches for gather actor according the funcgraph in partial.
  mindspore::HashMap<FuncGraph *, std::vector<AID>> output_data_with_branch_id_arrows_;
  // The real index of actor output, the bool value means if the output is a dynamic len.
  // eg. argument: (A, (B1, B2), C)  parameter: (a, b, c) the vector would be {<{0}, false>, <{1, 2}, true>,
  // <{3}, false>}
  mindspore::HashMap<FuncGraph *, std::vector<std::pair<std::vector<size_t>, bool>>> dynamic_len_index_;
};

using GatherActorPtr = std::shared_ptr<GatherActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_GATHER_ACTOR_H_
