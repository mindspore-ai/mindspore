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
#include "runtime/framework/actor/memory_interface_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/device_tensor_store.h"
#include "backend/kernel_compiler/kernel.h"
#include "ir/anf.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;

// The kernel actor is used to receive the device tensors and control info to luanch kernel.
// The processing flow is RunOpData/RunOpControl -> CheckLaunchCondition -> AllocateMemory
// -> OnMemoryAllocFinish -> LaunchKernel -> SendOutput -> FreeMemory.
class KernelActor : public MemoryInterfaceActor {
 public:
  KernelActor(std::string name, CNodePtr kernel, const DeviceContext *device_context, const AID memory_manager_aid)
      : MemoryInterfaceActor(name),
        kernel_(kernel),
        device_context_(device_context),
        memory_manager_aid_(memory_manager_aid),
        input_datas_num_(0),
        input_controls_num_(0) {}
  ~KernelActor() override = default;

  // The kernel actor run when receive the input data.
  void RunOpData(OpDataPtr<DeviceTensor> input_data, OpContext<DeviceTensor> *context) override;
  // The kernel actor run when receive the input control.
  void RunOpControl(AID *input_control, OpContext<DeviceTensor> *context) override;

  // The memory related operation interface.
  void AllocateMemory(OpContext<DeviceTensor> *context) override;
  void FreeMemory(OpContext<DeviceTensor> *context) override;
  // The real kernel launch processing after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *context) override;

 private:
  friend class GraphScheduler;

  // Check whether satisfy the condition for launch.
  bool CheckLaunchCondition(OpContext<DeviceTensor> *context) const;
  // Fetch the args of kernel launch.
  void FetchLaunchArgs(std::vector<AddressPtr> *kernel_inputs, std::vector<AddressPtr> *kernel_outputs,
                       std::vector<AddressPtr> *kernel_workspaces) const;
  // Send output data and output controls when finish kernel launch.
  void SendOutput(OpContext<DeviceTensor> *context) const;
  // Erase input data and input controls when finish kernel launch.
  void EraseInput(OpContext<DeviceTensor> *context);

  // Fetch the device tensor for launch.
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *context);
  void FetchOutputDeviceTensor();
  void FetchWorkspaceDeviceTensor();

  CNodePtr kernel_;
  // The device interface of kernel launch.
  const DeviceContext *device_context_;

  // The id of memory manager actor. Send message to it for alloc and free memory during the kernel launch.
  const AID memory_manager_aid_;

  // The dependent input data number.
  size_t input_datas_num_;
  // The dependent input controls number.
  size_t input_controls_num_;

  // Pair<index, anfNode> points to the dependent device tensor store, anfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, void *>> device_tensor_store_keys_;

  // The device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
  std::vector<DeviceTensor *> output_device_tensors_;
  std::vector<DeviceTensor *> workspace_device_tensors_;
};

using KernelActorPtr = std::shared_ptr<KernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
