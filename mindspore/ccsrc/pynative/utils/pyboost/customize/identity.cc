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

#include "include/pynative/utils/pyboost/customize/identity.h"
#include <memory>
#include <utility>
#include "ir/dtype/tensor_type.h"
#include "utils/stream_guard.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

void IdentityCustomizeCallWithoutContigous(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor]() {
    MS_LOG(DEBUG) << "Run device task Identity start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    auto input_x_address = std::dynamic_pointer_cast<device::DeviceAddress>(x_tensor->device_address());

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor);

    // Malloc for output tensors
    auto launch_kernel_tensor = runtime::DeviceAddressUtils::CreateKernelTensor(
      op->device_context(), outputs[0], x_tensor->storage_info()->ori_shape, op->stream_id());
    MS_EXCEPTION_IF_NULL(launch_kernel_tensor);
    auto launch_device_address = launch_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(launch_device_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                   launch_device_address->GetSize(), launch_device_address.get());
    if (!device_context->device_res_manager_->AllocateMemory(launch_device_address.get(), CurrentStream::id())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }

    // Get inputs kernel tensors, the not-tensor value will malloc here
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), x_tensor);
    if (input_address_info.first.empty()) {
      MS_LOG(EXCEPTION) << "Empty input kernel tensors.";
    }
    MS_EXCEPTION_IF_NULL(input_address_info.first[0]);
    if (!input_address_info.first[0]->host_info_exist()) {
      input_address_info.first[0]->SetHostInfo(std::make_shared<abstract::TensorShape>(x_tensor->shape()),
                                               std::make_shared<TensorType>(x_tensor->Dtype()), nullptr);
    }
    // Get outputs kernel tensors
    std::vector<kernel::KernelTensor *> output_kernel_tensor_list{launch_kernel_tensor.get()};
    std::vector<kernel::KernelTensorPtr> output_kernel_tensor_ptr_list{launch_kernel_tensor};
    const auto &output_address_info = std::make_pair(output_kernel_tensor_list, output_kernel_tensor_ptr_list);

    PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info,
                               op->stream_id());
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(outputs[0]->device_address());
    output_address->set_ptr(launch_device_address->GetMutablePtr());
    MS_LOG(DEBUG) << "Run device task Identity end";
  }));
}

void IdentityCustomizeCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor]() {
    MS_LOG(DEBUG) << "Run device task Identity start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();

    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    // Get inputs kernel tensors, the not-tensor value will malloc here
    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), x_tensor);
    if (input_address_info.first.empty()) {
      MS_LOG(EXCEPTION) << "Empty input kernel tensors.";
    }
    MS_EXCEPTION_IF_NULL(input_address_info.first[0]);
    if (!input_address_info.first[0]->host_info_exist()) {
      input_address_info.first[0]->SetHostInfo(std::make_shared<abstract::TensorShape>(x_tensor->shape()),
                                               std::make_shared<TensorType>(x_tensor->Dtype()), nullptr);
    }

    // Get outputs kernel tensors
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);
    if (output_address_info.first.empty()) {
      MS_LOG(EXCEPTION) << "Empty output kernel tensors.";
    }
    MS_EXCEPTION_IF_NULL(output_address_info.first[0]);
    if (!output_address_info.first[0]->host_info_exist()) {
      output_address_info.first[0]->SetHostInfo(std::make_shared<abstract::TensorShape>(outputs[0]->shape()),
                                                std::make_shared<TensorType>(outputs[0]->Dtype()), nullptr);
    }
    PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info,
                               op->stream_id());
    MS_LOG(DEBUG) << "Run device task Identity end";
  }));
}

tensor::TensorPtr IdentityCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  OpRunner::InferOpOutput(op, x_tensor);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  IdentityCall(op, x_tensor);
  return op->output(0);
}

void IdentityCall(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  if (x_tensor->is_contiguous()) {
    MS_LOG(DEBUG) << "Run Identity input contiguous";
    IdentityCustomizeCall(op, x_tensor);
  } else {
    MS_LOG(DEBUG) << "Run Identity input without contiguous";
    IdentityCustomizeCallWithoutContigous(op, x_tensor);
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
