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

#include "mindspore/ccsrc/kernel/pyboost/customize/identity.h"
#include <memory>
#include <utility>

namespace mindspore {
namespace kernel {
namespace pyboost {

void IdentityCustomizeCallWithoutContigous(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor,
                                           void *stream) {
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor, stream]() {
    MS_LOG(DEBUG) << "Run device task Identity start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    auto input_x_address = std::dynamic_pointer_cast<device::DeviceAddress>(x_tensor->device_address());

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor);

    // Malloc for output tensors
    auto launch_device_address = runtime::DeviceAddressUtils::CreateDeviceAddress(
      op->device_context(), outputs[0], x_tensor->storage_info()->ori_shape, op->stream_id());
    if (!device_context->device_res_manager_->AllocateMemory(launch_device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }

    // Get inputs kernel tensors, the not-tensor value will malloc here
    const auto &input_address_info = PyBoostUtils::GetAddressInfo(device_context, op->input_abs(), x_tensor);

    // Get outputs kernel tensors
    std::vector<kernel::KernelTensor *> output_kernel_tensor_list{launch_device_address->kernel_tensor().get()};
    device::DeviceAddressPtrList output_device_address_list{launch_device_address};
    const auto &output_address_info = std::make_pair(output_kernel_tensor_list, output_device_address_list);

    PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info, stream);
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(outputs[0]->device_address());
    output_address->SetStorageInfo(input_x_address->GetStorageInfo());
    output_address->set_ptr(launch_device_address->GetMutablePtr());
    MS_LOG(DEBUG) << "Run device task Identity end";
  }));
}

void IdentityCustomizeCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor, void *stream) {
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor, stream]() {
    MS_LOG(DEBUG) << "Run device task Identity start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    // Get inputs kernel tensors, the not-tensor value will malloc here
    const auto &input_address_info = PyBoostUtils::GetAddressInfo(device_context, op->input_abs(), x_tensor);

    // Get outputs kernel tensors
    const auto &output_address_info = PyBoostUtils::GetAddressInfo(device_context, {op->output_abs()}, outputs);

    PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info, stream);
    MS_LOG(DEBUG) << "Run device task Identity end";
  }));
}

tensor::TensorPtr IdentityCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor, void *stream) {
  OpRunner::InferOpOutput(op, x_tensor);

  PyBoostUtils::PrepareOpInputs(op->device_context(), x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->outputs());

  if (x_tensor->is_contiguous()) {
    MS_LOG(DEBUG) << "Run Identity input contiguous";
    IdentityCustomizeCall(op, x_tensor, stream);
  } else {
    MS_LOG(DEBUG) << "Run Identity input without contiguous";
    IdentityCustomizeCallWithoutContigous(op, x_tensor, stream);
  }
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
