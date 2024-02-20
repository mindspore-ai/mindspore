/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/pyboost/customize/identity.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "transform/acl_ir/acl_helper.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

void IdentityCustomizeCallWithoutContigous(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor]() {
    MS_LOG(DEBUG) << "Run device task Identity start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    auto input_shape = x_tensor->storage_info()->ori_shape;
    const auto &output_shape = x_tensor->storage_info()->ori_shape;
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor);
    // Malloc for output tensors
    auto launch_device_address = runtime::DeviceAddressUtils::CreateDeviceAddress(
      op->device_context(), outputs[0], x_tensor->storage_info()->ori_shape, op->stream_id());
    if (!device_context->device_res_manager_->AllocateMemory(launch_device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }

    auto identity_kernel = std::make_shared<kernel::AclKernelMod>();
    auto input_x_address = std::dynamic_pointer_cast<device::DeviceAddress>(x_tensor->device_address());

    if (!input_x_address->kernel_tensor()->host_info_exist()) {
      input_x_address->kernel_tensor()->SetHostInfo(std::make_shared<abstract::TensorShape>(x_tensor->shape()),
                                                    std::make_shared<TensorType>(x_tensor->Dtype()), nullptr);
    }
    if (!launch_device_address->kernel_tensor()->host_info_exist()) {
      launch_device_address->kernel_tensor()->SetHostInfo(std::make_shared<abstract::TensorShape>(output_shape),
                                                          std::make_shared<TensorType>(outputs[0]->Dtype()), nullptr);
    }
    auto input_kernel_tensors = {input_x_address->kernel_tensor().get()};
    auto output_kernel_tensors = {launch_device_address->kernel_tensor().get()};

    if (!std::static_pointer_cast<KernelMod>(identity_kernel)
           ->Init(prim::kPrimIdentity, input_kernel_tensors, output_kernel_tensors)) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Initialize acl kernel op[Identity] failed.";
    }
    identity_kernel->CreateAclConverter();
    identity_kernel->SetDeviceInfo({input_x_address->format()}, {launch_device_address->format()},
                                   {input_x_address->type_id()}, {launch_device_address->type_id()});

    identity_kernel->PackageInput(kIndex0, input_x_address->format(), &input_shape);
    identity_kernel->PackageOutput(kIndex0, output_shape);
    identity_kernel->SetNeedConvertHostTensor(true);

    if (identity_kernel->Resize(input_kernel_tensors, output_kernel_tensors) != KRET_OK) {
      MS_LOG(EXCEPTION) << "Kernel identity resize failed";
    }
    auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());

    auto workspace_address = PyBoostUtils::CreateWorkSpaceDeviceAddress(identity_kernel, device_context, "Identity");
    auto workspaces = PyBoostUtils::GetKernelTensorFromAddress(workspace_address);

    if (!identity_kernel->Launch(input_kernel_tensors, workspaces, output_kernel_tensors, stream_ptr)) {
      MS_LOG(EXCEPTION) << "Launch kernel identity failed";
    }
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(outputs[0]->device_address());
    output_address->SetStorageInfo(input_x_address->GetStorageInfo());
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
    auto input_shape = x_tensor->shape();
    auto output_shape = outputs[0]->shape();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    auto identity_kernel = std::make_shared<kernel::AclKernelMod>();
    auto input_x_address = std::dynamic_pointer_cast<device::DeviceAddress>(x_tensor->device_address());
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(outputs[0]->device_address());

    if (!input_x_address->kernel_tensor()->host_info_exist()) {
      input_x_address->kernel_tensor()->SetHostInfo(std::make_shared<abstract::TensorShape>(x_tensor->shape()),
                                                    std::make_shared<TensorType>(x_tensor->Dtype()), nullptr);
    }
    if (!output_address->kernel_tensor()->host_info_exist()) {
      output_address->kernel_tensor()->SetHostInfo(std::make_shared<abstract::TensorShape>(output_shape),
                                                   std::make_shared<TensorType>(outputs[0]->Dtype()), nullptr);
    }
    auto input_kernel_tensors = {input_x_address->kernel_tensor().get()};
    auto output_kernel_tensors = {output_address->kernel_tensor().get()};

    if (!std::static_pointer_cast<KernelMod>(identity_kernel)
           ->Init(prim::kPrimIdentity, input_kernel_tensors, output_kernel_tensors)) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Initialize acl kernel op[Identity] failed.";
    }
    identity_kernel->CreateAclConverter();
    identity_kernel->SetDeviceInfo({input_x_address->format()}, {output_address->format()},
                                   {input_x_address->type_id()}, {output_address->type_id()});

    identity_kernel->PackageInput(kIndex0, input_x_address->format(), &input_shape);
    identity_kernel->PackageOutput(kIndex0, output_shape);
    identity_kernel->SetNeedConvertHostTensor(true);

    if (identity_kernel->Resize(input_kernel_tensors, output_kernel_tensors) != KRET_OK) {
      MS_LOG(EXCEPTION) << "Kernel identity resize failed";
    }
    auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());

    auto workspace_address = PyBoostUtils::CreateWorkSpaceDeviceAddress(identity_kernel, device_context, "Identity");
    auto workspaces = PyBoostUtils::GetKernelTensorFromAddress(workspace_address);

    if (!identity_kernel->Launch(input_kernel_tensors, workspaces, output_kernel_tensors, stream_ptr)) {
      MS_LOG(EXCEPTION) << "Launch kernel identity failed";
    }
    MS_LOG(DEBUG) << "Run device task Identity end";
  }));
}

tensor::TensorPtr IdentityAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  OpRunner::InferOpOutput(op, x_tensor);

  PyBoostUtils::PrepareOpInputs(op->device_context(), x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->outputs());

  if (x_tensor->is_contiguous()) {
    MS_LOG(DEBUG) << "Run Identity input contiguous";
    IdentityCustomizeCall(op, x_tensor);
  } else {
    MS_LOG(DEBUG) << "Run Identity input without contiguous";
    IdentityCustomizeCallWithoutContigous(op, x_tensor);
  }
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
