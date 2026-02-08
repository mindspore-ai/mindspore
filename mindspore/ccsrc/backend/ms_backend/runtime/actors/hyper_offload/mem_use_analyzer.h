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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEM_USE_ANALYZER_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEM_USE_ANALYZER_ACTOR_H_

#include <memory>
#include <unordered_map>

#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"
#include "backend/ms_backend/runtime/actors/base/super_kernel_actor.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/mem_counted_cache.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace runtime {
using MemCountedCachePtr = std::shared_ptr<MemCountedCache>;

class MemUseAnalyzer {
 public:
  MemUseAnalyzer(const MemUseAnalyzer &) = delete;
  MemUseAnalyzer &operator=(const MemUseAnalyzer &) = delete;
  ~MemUseAnalyzer() = default;

  static MemUseAnalyzer &GetInstance() {
    static MemUseAnalyzer instance;
    return instance;
  }

  void InitGraphInfo(SuperKernelActor *super_actor, const device::DeviceContext *device_context);

  void LaunchTaskBefore(KernelRunner *kernel_actor, const device::DeviceContext *device_context,
                        bool need_check_output_mem = true);

  void LaunchTaskAfter(KernelRunner *kernel_actor, const device::DeviceContext *device_context);

  std::vector<std::pair<KernelTensorPtr, size_t>> GetDeviceKernelTensorsInfo(size_t idx,
                                                                             const KernelTensorPtrList &kernel_tensors,
                                                                             bool need_pop_user);

  void ClearGraphInfo(const device::DeviceContext *device_context);

 private:
  MemUseAnalyzer() : mem_counted_cache_{std::make_shared<MemCountedCache>()} {};

  void MarkGraphIndex(const std::vector<KernelRunnerPtr> &kernel_actors);

  std::vector<std::pair<KernelTensorPtr, size_t>> GetKernelTensorUserInfo(size_t idx, KernelRunner *kernel_actor,
                                                                          bool need_output = false,
                                                                          bool need_pop_user = true);
  void UpdateCopyKernelTensors(const KernelTensorPtrPairList &kernel_tensors_pair);
  void RefreshInputKernelTensors(KernelRunner *kernel_actor);
  KernelTensorPtr FindDeviceKernelTensor(const KernelTensorPtr &kernel_tensor);
  void ProcessGraphOutputLaunch(const device::DeviceContext *device_context, size_t stream_id);

  MemCountedCachePtr mem_counted_cache_{nullptr};

  // Original kernel tensors map user_indexes
  mindspore::HashMap<KernelTensorPtr, std::queue<size_t>> kernel_tensor_info_;
  // Original kernel tensors map copied kernel tensors
  mindspore::HashMap<KernelTensorPtr, KernelTensorPtr> original_tensors_copyed_map_;
  // Copied kernel tensors map original kernel tensors
  mindspore::HashMap<KernelTensorPtr, KernelTensorPtr> copyed_tensors_original_map_;
  // KernelActor map idx
  mindspore::HashMap<KernelRunner *, size_t> kernel_actor_idx_map_;
  std::vector<KernelTensorPtr> output_kernel_tensors_;
  SuperKernelActor *super_kernel_actor_;
  size_t copy_out_stream_id_;
  size_t copy_in_stream_id_;
  size_t max_idx_;

  // ConditionSwitch info
  mindspore::HashMap<KernelRunner *, ConditionSwitchBranchInfoPtr> switch_info_map_;
  std::vector<ConditionSwitchBranchInfoPtr> latest_switch_infos_;
  std::vector<KernelRunnerPtr> kernel_actors_;
  size_t old_horizon_;
};

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEM_USE_ANALYZER_ACTOR_H_
