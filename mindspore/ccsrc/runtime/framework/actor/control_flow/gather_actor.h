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
#include <unordered_map>
#include <stack>
#include <utility>
#include <algorithm>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/abstract_actor.h"

namespace mindspore {
namespace runtime {

// Gather actor will be used in the control flow. When the subgraph is called, the real parameters need to be put
// together and sent to the subgraph.
class GatherActor : public AbstractActor {
 public:
  GatherActor(const std::string &name, const std::vector<KernelWithIndex> &parameters)
      : AbstractActor(name, KernelTransformType::kGatherActor, nullptr), formal_parameters_(parameters) {}
  ~GatherActor() override = default;

  // The gather actor collects single node when receive the result of kernel actor.
  void CollectRealParameter(const AnfNodePtr &node, size_t index, size_t position,
                            OpContext<DeviceTensor> *const context);
  // The gather actor collects all real parameters when receive the output of switch actor.
  void CollectRealParameters(const std::vector<KernelWithIndex> &real_parameters, size_t position,
                             OpContext<DeviceTensor> *const context);

 private:
  friend class GraphScheduler;

  // Formal parameters of actor, which is the front node.
  std::vector<KernelWithIndex> formal_parameters_;

  // Input data.
  std::unordered_map<uuids::uuid *, std::unordered_map<size_t, std::vector<KernelWithIndex>>> input_nodes_;
  // The store node records the value node input of the gather actor.
  std::vector<std::pair<size_t, KernelWithIndex>> store_nodes_;

  // Output arrow.
  std::unordered_map<AnfNodePtr, std::pair<AID, size_t>> output_branch_arrows_;
};

using GatherActorPtr = std::shared_ptr<GatherActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_GATHER_ACTOR_H_
