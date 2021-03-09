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
#include "mindrt/include/actor/op_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/device_tensor_store.h"
#include "backend/kernel_compiler/kernel.h"
#include "ir/anf.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::kernel::AddressPtr;

// The kernel actor is used to receive the device tensors and control info to luanch kernel.
class KernelActor : public OpActor<DeviceTensor> {
 public:
  KernelActor(std::string name, CNodePtr kernel, const DeviceContext *device_context)
      : OpActor(name), kernel_(kernel), device_context_(device_context), input_datas_num_(0), input_controls_num_(0) {}
  virtual ~KernelActor() = default;

  // The kernel actor run when receive the input data.
  void RunOpData(OpDataPtr<DeviceTensor> input_data, OpContext<DeviceTensor> *context) override;
  // The kernel actor run when receive the input control.
  void RunOpControl(AID *input_control, OpContext<DeviceTensor> *context) override;

 private:
  friend class GraphScheduler;

  // Check whether satisfy the condition for launch.
  bool CheckLaunchCondition(const uuids::uuid *sequential_num);
  // Fetch the args of kernel launch.
  void FetchLaunchArgs(std::vector<AddressPtr> *kernel_inputs, std::vector<AddressPtr> *kernel_outputs,
                       std::vector<AddressPtr> *kernel_workspaces);
  // The real kernel launch processing.
  void Launch(OpContext<DeviceTensor> *context);
  // Send output data and output controls when finish kernel launch.
  void SendOutput(OpContext<DeviceTensor> *context);

  void AllocateMemory(OpContext<DeviceTensor> *context);
  void FreeMemory(OpContext<DeviceTensor> *context);

  // Fetch the device tensor for launch.
  void FetchInputDeviceTensor(const uuids::uuid *sequential_num);
  void FetchOutputDeviceTensor();
  void FetchWorkspaceDeviceTensor();

  CNodePtr kernel_;
  // The device interface of kernel launch.
  const DeviceContext *device_context_;

  // The dependent input data number.
  size_t input_datas_num_;
  // The dependent input controls number.
  size_t input_controls_num_;

  // Pair<index, anfNode> points to the dependent device tensor store, anfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, void *>> device_tensor_store_keys_;

  // The device tensors for launch.
  std::vector<DeviceTensorPtr> input_device_tensors_;
  std::vector<DeviceTensorPtr> output_device_tensors_;
  std::vector<DeviceTensorPtr> workspace_device_tensors_;
};

using KernelActorPtr = std::shared_ptr<KernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
