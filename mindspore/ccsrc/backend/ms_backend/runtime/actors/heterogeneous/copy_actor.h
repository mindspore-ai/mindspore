/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_COPY_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_COPY_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <set>

#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "backend/ms_backend/runtime/actors/base/memory_aware_actor.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/kernel_base/device_tensor_store.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;

// The copy actor is used to receive the device tensors and control info to copy data between input device tensor and
// output device tensor. The processing flow is RunOpData/RunOpControl -> CheckRunningCondition -> SendMemoryAllocReq
// -> OnMemoryAllocFinish -> Copy -> SendMemoryFreeReq -> SendOutput.
class CopyActor : public MemoryAwareActor {
 public:
  CopyActor(const std::string &name, AnfNode *from_kernel, const KernelGraphPtr &from_graph,
            const AID &memory_manager_aid)
      : MemoryAwareActor(name, KernelTransformType::kCopyActor, nullptr, memory_manager_aid),
        from_kernel_(from_kernel),
        from_graph_(from_graph),
        output_(nullptr),
        is_need_update_output_size_(false) {}
  ~CopyActor() override = default;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<KernelTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<KernelTensor> *const context) override;
  // The copy processing after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<KernelTensor> *const context) override;

  const KernelTensorPtr &output() const { return output_; }
  bool is_need_update_output_size() const { return is_need_update_output_size_; }

 protected:
  void Init() override;
  void Run(OpContext<KernelTensor> *const context) override;
  void UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &data_arrow,
                        const AnfNodePtr &output_node, OpContext<KernelTensor> *const context) override;
  void IncreaseNewRefCounts(OpContext<KernelTensor> *const context) override;

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;

  // Fetch the device tensor for copy.
  void FetchKernelTensor(OpContext<KernelTensor> *const context);

  // The copy source.
  AnfNode *from_kernel_;
  KernelGraphPtr from_graph_;

  // The input device tensor is saved from the input data or fetched by device_tensor_store_keys_.
  std::vector<KernelTensorPtr> input_kernel_tensors_;
  // The output device tensor is saved from the output or fetched by device_tensor_store_keys_.
  std::vector<KernelTensorPtr> output_kernel_tensors_;

  KernelTensorPtr output_;
  // If to actor does not need the output the address, such as "only shape depend" of kernel actor or "is not used" of
  // super kernel actor, the output ref count needs to be released when sending output.
  size_t output_free_size_{0};
  // The output size needs to be updated in the dynamic shape scene.
  bool is_need_update_output_size_;
  // The ref internal parameter device address of the copy actor output.
  std::set<DeviceTensorPtr> ref_parameter_device_tensors_;
};

using CopyActorPtr = std::shared_ptr<CopyActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_COPY_ACTOR_H_
