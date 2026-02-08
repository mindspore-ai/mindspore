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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_SWITCH_RUNNER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_SWITCH_RUNNER_H_

#include <set>
#include <string>
#include <vector>
#include <memory>
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;

// Condition switch actor is used to execute the branch according to the input condition in kernel graph.
class ConditionSwitchRunner : public KernelRunner {
 public:
  ConditionSwitchRunner(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                        const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                        GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                        const std::set<size_t> &modifiable_ref_output_indexes,
                        const KernelTransformType &type = KernelTransformType::kConditionSwitchActor);
  ~ConditionSwitchRunner() override = default;
  const std::vector<std::string> &branch_names() const { return branch_names_; }
  const mindspore::HashMap<std::string, std::vector<size_t>> &branch_output_free_index() const {
    return branch_output_free_index_;
  }
  std::shared_ptr<bool[]> GetBranchFlag() { return branch_flags_; }

 protected:
  void Init() override;
  void UpdateRefDeviceAddress(OpContext<KernelTensor> *const context, bool increase_ref_count) override;
  void ExecuteInferShapeTask(OpContext<KernelTensor> *const context, bool high_perf) override;
  void ExecuteResizeKernelModTask(OpContext<KernelTensor> *const context, bool high_perf) override;
  void ExecuteLaunchKernelTask(OpContext<KernelTensor> *const context) override;
  void ExecuteLaunchKernelTaskHP(OpContext<KernelTensor> *const context) override;
  void UpdateMemoryFreeList(OpContext<KernelTensor> *const context) override;

 private:
  void FetchParameterInput(OpContext<KernelTensor> *const context);

  friend class SuperKernelActor;
  // Collect memory free list, as the ref counts of different branches are superimposed on the output,
  // so the excess reference counts of other branches need to be subtracted in advance.
  void CollectMemoryFreeList(size_t index);

  // Graph name of each branch,
  std::vector<std::string> branch_names_;
  // Ref count of each branch.
  std::vector<std::vector<size_t>> branch_origin_ref_count_;
  // Branch of data arrow and control arrow.
  std::vector<size_t> output_data_branch_indexes_;
  std::vector<size_t> output_control_branch_indexes_;

  // Cache output data by output index to modify the output data effectively.
  std::vector<std::vector<OpData<DeviceTensor> *>> output_data_by_output_index_;

  // Switch needs to send current branch name to the corresponding gather actor to check its inputs.
  AID *gather_aid_{nullptr};

  // The pointer points to the current branch name of condition gather actor to inform it the enable branch.
  std::string *gather_branch_name_{nullptr};
  // The enable flag of each branch to control the kernel actor between the condition actors.
  std::shared_ptr<bool[]> branch_flags_;
  // The output free index of each branch.
  mindspore::HashMap<std::string, std::vector<size_t>> branch_output_free_index_;
};

using ConditionSwitchRunnerPtr = std::shared_ptr<ConditionSwitchRunner>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CONTROLFLOW_CONDITION_SWITCH_RUNNER_H_
