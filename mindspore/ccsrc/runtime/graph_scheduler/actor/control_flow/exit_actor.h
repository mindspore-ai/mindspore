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
#include <utility>
#include "utils/hash_map.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/control_flow/control_actor.h"

namespace mindspore {
namespace runtime {
// The exit actor is used to receive a set of data arrow and a branch id in the control flow, and then send the
// device tensors in the data to the corresponding actor. It is the exit of the end of kernel graph execution.
class ExitActor : public ControlActor {
 public:
  ExitActor(const std::string &name, const AID &memory_manager_aid, const std::vector<KernelWithIndex> &parameters,
            const AnfNodePtr &node)
      : ControlActor(name, KernelTransformType::kExitActor, memory_manager_aid, parameters, node) {
    device_contexts_.resize(parameters.size());
    input_device_tensors_.resize(parameters.size());
  }
  ~ExitActor() override = default;

  const mindspore::HashMap<int, std::vector<AID>> &output_branch_control_arrows() const {
    return output_branch_control_arrows_;
  }
  const mindspore::HashMap<int, std::vector<DataArrowPtr>> &output_branch_data_arrows() const {
    return output_branch_data_arrows_;
  }
  const mindspore::HashMap<int, std::vector<DataArrowPtr>> &output_branch_partial_arrows() const {
    return output_branch_partial_arrows_;
  }
  const std::vector<bool> &is_need_copy_device_tensors() const { return is_need_copy_device_tensors_; }
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

 protected:
  void Init() override;
  void FetchInput(OpContext<DeviceTensor> *const context) override;
  void SendOutput(OpContext<DeviceTensor> *const context) override;
  void IncreaseDynamicRefCounts(OpContext<DeviceTensor> *const context) override;

 private:
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;

  void CopyDeviceAddress(OpContext<DeviceTensor> *const context);

  // Exit actor will send to different actors according to different callers, so the output data, control,
  // and partial arrows will have branch.
  mindspore::HashMap<int, std::vector<DataArrowPtr>> output_branch_data_arrows_;
  mindspore::HashMap<int, std::vector<AID>> output_branch_control_arrows_;
  mindspore::HashMap<int, std::vector<DataArrowPtr>> output_branch_partial_arrows_;

  // The exit actor needs to create a new device address and take out the ptr from the device tensor come from
  // the kernel actor. These new created device tensors are stored in the created device tensors.
  std::vector<DeviceTensorPtr> created_device_tensors_;
  // In exit actor, we need to copy a new device tensor for the output of the kernel actor, but parameter is not
  // needed. This mark is used to record whether it need to be copied.
  std::vector<bool> is_need_copy_device_tensors_;
  // Cache the dynamic shape flag to optimize the running performance.
  std::vector<bool> is_dynamic_shapes_;
  // Output data.
  //  The output branch data corresponds to the output_data_arrows_ one by one.
  mindspore::HashMap<int, std::vector<std::pair<size_t, OpDataUniquePtr<DeviceTensor>>>> output_branch_data_;
};

using ExitActorPtr = std::shared_ptr<ExitActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_EXIT_ACTOR_H_
