/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_GATHER_RUNNER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_GATHER_RUNNER_H_

#include <set>
#include <vector>
#include <string>
#include <memory>
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;

// Condition gather actor is used to collect the output of different branch from condition switch actor.
class ConditionGatherRunner : public KernelRunner {
 public:
  ConditionGatherRunner(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                        const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                        GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                        const std::set<size_t> &modifiable_ref_output_indexes,
                        const KernelTransformType &type = KernelTransformType::kConditionGatherActor);
  ~ConditionGatherRunner() override;
  void ExecuteInferShapeTask(OpContext<KernelTensor> *const context, bool high_perf) override;
  void ExecuteResizeKernelModTask(OpContext<KernelTensor> *const context, bool high_perf) override;
  void ExecuteLaunchKernelTask(OpContext<KernelTensor> *const context) override;
  void ExecuteLaunchKernelTaskHP(OpContext<KernelTensor> *const context) override;
  void UpdateRefDeviceAddress(OpContext<KernelTensor> *const context, bool increase_ref_count) override;
  size_t branch_output_num() const { return branch_output_num_; }
  const std::vector<std::string> &branch_names() const { return branch_names_; }

 protected:
  void Init() override;

 private:
  void FetchParameterInput(size_t start_index, OpContext<DeviceTensor> *const context);

  friend class SuperKernelActor;
  // Output num of each branch.
  size_t branch_output_num_{0};
  // The order of each branch name.
  std::vector<std::string> branch_names_;
  // The current execute branch between switch and gather actor.
  std::string current_branch_name_;
  // Input data and control num for each branch.
  mindspore::HashMap<std::string, size_t> branch_name_to_id_;
  mindspore::HashMap<std::string, size_t> branch_name_to_input_data_num_;
  mindspore::HashMap<std::string, size_t> branch_name_to_input_control_num_;
  std::vector<device::DeviceAddressPtr> need_clean_ptr_device_addresses_;
  std::shared_ptr<bool[]> branch_flags_;
};

using ConditionGatherRunnerPtr = std::shared_ptr<ConditionGatherRunner>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_GATHER_RUNNER_H_
