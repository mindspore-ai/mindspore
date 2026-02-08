/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_EXIT_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_EXIT_ACTOR_H_

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <utility>

#include "utils/hash_map.h"
#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/runtime/actor/control_flow/control_actor.h"

namespace mindspore {
namespace ge_backend {

namespace runtime {
// The exit actor is used to receive a set of data arrow and a branch id in the control flow, and then send the
// device tensors in the data to the corresponding actor. It is the exit of the end of kernel graph execution.
class ExitActor : public ControlActor {
 public:
  ExitActor(const std::string &name, const AID &memory_manager_aid, const std::vector<KernelWithIndex> &parameters,
            const AnfNodePtr &node)
      : ControlActor(name, KernelTransformType::kExitActor, memory_manager_aid, parameters, node) {
    input_kernel_tensors_.resize(parameters.size());
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
  const mindspore::HashMap<int, std::vector<std::pair<std::vector<size_t>, bool>>> &output_branch_dynamic_len_index()
    const {
    return output_branch_dynamic_len_index_;
  }
  void OnMemoryAllocFinish(OpContext<KernelTensor> *const context) override;

 protected:
  void Init() override;
  void FetchInput(OpContext<KernelTensor> *const context) override;
  void SendOutput(OpContext<KernelTensor> *const context) override;
  void IncreaseDynamicRefCounts(OpContext<KernelTensor> *const context) override;

 private:
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;

  void CopyDeviceAddress(OpContext<KernelTensor> *const context);
  void UpdateDeviceOutputData();
  bool IsNeedCopyDeviceAddress(const KernelTensorPtr &input_kernel_tensor, size_t index);

  // Exit actor will send to different actors according to different callers, so the output data, control,
  // and partial arrows will have branch.
  mindspore::HashMap<int, std::vector<DataArrowPtr>> output_branch_data_arrows_;
  mindspore::HashMap<int, std::vector<AID>> output_branch_control_arrows_;
  mindspore::HashMap<int, std::vector<DataArrowPtr>> output_branch_partial_arrows_;
  // The real index of actor output, the first int means the output branch id and the bool value means if the
  // output is a dynamic len.
  // eg. argument: (A, (B1, B2), C)  parameter: (a, b, c)
  //     the vector would be {<{0}, false>, <{1, 2}, true>,<{3},false>}
  mindspore::HashMap<int, std::vector<std::pair<std::vector<size_t>, bool>>> output_branch_dynamic_len_index_;

  // In exit actor, we need to copy a new device tensor for the output of the kernel actor, but parameter is not
  // needed. This mark is used to record whether it need to be copied.
  std::vector<bool> is_need_copy_device_tensors_;
  std::vector<bool> is_need_dynamic_checks_;
  std::map<KernelWithIndex, KernelWithIndex> ref_out_in_map_;
  // Cache the dynamic shape flag to optimize the running performance.
  std::vector<bool> is_dynamic_shapes_;
  // Output data.
  //  The output branch data corresponds to the output_data_arrows_ one by one.
  mindspore::HashMap<int, std::vector<std::pair<size_t, OpDataUniquePtr<KernelTensor>>>> output_branch_data_;
  // The value of haspmap indicates the output data flag. See constant prefixed with kOutputDataFalg for details.
  mindspore::HashMap<int, std::vector<size_t>> output_branch_data_flag_;
};

using ExitActorPtr = std::shared_ptr<ExitActor>;
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_EXIT_ACTOR_H_
