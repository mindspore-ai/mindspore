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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_SWITCH_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_SWITCH_ACTOR_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <utility>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/abstract_actor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;

constexpr size_t kSwitchCondPos = 1;
constexpr size_t kMaxSwitchCondSize = 8;

// Switch actor is used to execute the branch according to the input condition.
// Switch and SwitchLayer node will be converted to switch actor.
class SwitchActor : public AbstractActor {
 public:
  SwitchActor(const std::string &name, const std::vector<KernelWithIndex> &parameters)
      : AbstractActor(name, KernelTransformType::kSwitchActor, nullptr), formal_parameters_(parameters) {
    input_result_num_ = formal_parameters_.size();
  }
  ~SwitchActor() override = default;

  void Init() override;

  // The switch actor collects single node when receive the result of kernel actor.
  void CollectRealParameter(const AnfNodePtr &node, size_t index, size_t position,
                            OpContext<DeviceTensor> *const context);
  // The switch actor collects all real parameters when receive the output of gather actor.
  void CollectRealParameters(const std::vector<KernelWithIndex> &real_parameters, size_t position,
                             OpContext<DeviceTensor> *const context);

 private:
  friend class GraphScheduler;
  size_t GetIndex(const OpContext<DeviceTensor> *const context);

  // Formal parameters of actor, which is the front node.
  std::vector<KernelWithIndex> formal_parameters_;

  // Input data.
  std::unordered_map<uuids::uuid *, std::unordered_map<size_t, std::vector<KernelWithIndex>>> input_nodes_;
  // The store node records the value node input of the switch actor.
  std::vector<std::pair<size_t, AnfNodePtr>> store_nodes_;

  // Output arrow.
  std::vector<std::vector<DataArrowPtr>> output_branch_data_arrows_;
  std::vector<std::vector<DataArrowPtr>> output_branch_result_arrows_;
  std::vector<AID> output_branch_real_parameter_arrows_;
  //  The output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<std::vector<OpDataUniquePtr<DeviceTensor>>> output_data_;
  size_t input_result_num_;
};

using SwitchActorPtr = std::shared_ptr<SwitchActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_SWITCH_ACTOR_H_
