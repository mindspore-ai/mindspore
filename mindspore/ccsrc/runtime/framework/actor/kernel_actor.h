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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/debug_aware_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/device_tensor_store.h"
#include "backend/kernel_compiler/kernel.h"
#include "ir/anf.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::device::KernelInfo;
using mindspore::kernel::Address;
using mindspore::kernel::KernelLaunchInfo;
using mindspore::tensor::TensorPtr;

// The kernel actor is used to receive the device tensors and control info to luanch kernel.
// The processing flow is RunOpData/RunOpControl -> CheckRunningCondition -> SendMemoryAllocReq
// -> OnMemoryAllocFinish -> LaunchKernel -> SendMemoryFreeReq -> SendOutput.
class KernelActor : public DebugAwareActor {
 public:
  KernelActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
              const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
              GraphExecutionStrategy strategy)
      : DebugAwareActor(name, KernelTransformType::kKernelActor, recorder_aid, memory_manager_aid, debug_aid),
        kernel_(kernel),
        kernel_info_(nullptr),
        is_dynamic_shape_(false),
        real_input_num_(0),
        strategy_(strategy) {
    (void)device_contexts_.emplace_back(device_context);
  }
  ~KernelActor() override = default;

  void Init() override;

  // The kernel actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) override;
  // The kernel actor run when receive the input control.
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;
  // The kernel actor run when receive the input control and input tensors, used in step mode.
  void RunOpControlWithInputTensor(AID *const input_control, OpContext<DeviceTensor> *const context,
                                   const std::vector<TensorPtr> *input_tensors);

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  // The debug related operation interface.
  void SendDebugReq(OpContext<DeviceTensor> *const context) override;
  // The callback after debug finished.
  void OnDebugFinish(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  // Fetch the device tensor for launch.
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context);
  void FetchOutputDeviceTensor();
  void CopyInputDeviceTensor(const OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *const context);
  // In step mode, push the input tensors which contain valid device address into input_device_tensors_ directly.
  void PushInputDeviceTensor(const std::vector<TensorPtr> *input_tensors);

  // The processing before kernel launch: update the info of kernel launch.
  void PreLaunchKernel(OpContext<DeviceTensor> *const context);
  // The processing after kernel launch: 1.erase input, 2.free memory, 3.send output.
  void PostLaunchKernel(OpContext<DeviceTensor> *const context);

  // Send output data and output controls when finish kernel launch.
  void SendOutput(OpContext<DeviceTensor> *const context) const;

  // The info of kernel.
  CNodePtr kernel_;
  KernelInfo *kernel_info_;
  bool is_dynamic_shape_;

  // The real input number of kernel launch.
  size_t real_input_num_;

  // The execution strategy of kernel actor.
  // In pipeline mode, kernel actor executes asynchronously.
  // In step mode, kernel actor executes synchronously.
  GraphExecutionStrategy strategy_{GraphExecutionStrategy::kPipeline};

  // The device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
  std::vector<DeviceTensor *> output_device_tensors_;
  std::vector<DeviceTensor *> workspace_device_tensors_;
  // The received input device type may be different from the device context type in the control flow and host device
  // scenarios, so it needs to be copied from the input device type to the device context type.
  std::vector<DeviceTensorPtr> copy_input_device_tensors_;

  // The device tensors for memory alloc and free.
  // output + workspace
  std::vector<DeviceTensor *> memory_alloc_list_;
  // input + output + workspace
  std::vector<DeviceTensor *> memory_free_list_;
  // The device tensor of external reference is not the real data of this kernel, but need add to the memory_free_list_.
  std::vector<DeviceTensor *> external_reference_tensors_;

  // The kernel launch info is fetched by the device tensors.
  KernelLaunchInfo launch_info_;

  // Cache unique output data by output index to modify the output data effectively.
  std::vector<std::vector<OpDataUniquePtr<DeviceTensor>>> output_data_by_output_index_;
  //  The output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<OpData<DeviceTensor> *> output_data_;
};

using KernelActorPtr = std::shared_ptr<KernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
