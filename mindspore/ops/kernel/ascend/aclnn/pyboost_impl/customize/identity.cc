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

#include "kernel/ascend/aclnn/pyboost_impl/customize/identity.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/customize/identity.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/aclop/kernel_mod_impl/acl_kernel_mod.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

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
    auto output_kernel_tensor = runtime::DeviceAddressUtils::CreateKernelTensor(
      op->device_context(), outputs[0], x_tensor->storage_info()->ori_shape, op->stream_id());
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto launch_device_address = output_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(launch_device_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                   launch_device_address->GetSize(), launch_device_address.get());
    if (!device_context->device_res_manager_->AllocateMemory(launch_device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }

    auto identity_kernel = std::make_shared<kernel::AclKernelMod>();
    const auto &input_x_address = x_tensor->device_address();
    auto abs = x_tensor->ToAbstract()->Broaden();
    MS_EXCEPTION_IF_NULL(abs);
    auto input_kernel_tensor = std::make_shared<KernelTensor>(abs->GetShape(), abs->GetType(), nullptr);
    input_kernel_tensor->set_device_address(input_x_address);
    input_kernel_tensor->set_tensor_storage_info(x_tensor->storage_info());
    input_kernel_tensor->SetShapeVector(x_tensor->shape());
    input_kernel_tensor->set_format(x_tensor->format());
    if (!input_kernel_tensor->host_info_exist()) {
      input_kernel_tensor->SetHostInfo(std::make_shared<abstract::TensorShape>(x_tensor->shape()),
                                       std::make_shared<TensorType>(x_tensor->Dtype()), nullptr);
    }

    if (!output_kernel_tensor->host_info_exist()) {
      output_kernel_tensor->SetHostInfo(std::make_shared<abstract::TensorShape>(output_shape),
                                        std::make_shared<TensorType>(outputs[0]->Dtype()), nullptr);
    }
    auto input_kernel_tensors = {input_kernel_tensor.get()};
    auto output_kernel_tensors = {output_kernel_tensor.get()};
    if (!std::static_pointer_cast<KernelMod>(identity_kernel)
           ->Init(prim::kPrimIdentity, input_kernel_tensors, output_kernel_tensors)) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Initialize acl kernel op[Identity] failed.";
    }
    identity_kernel->RefreshAclConverter(input_kernel_tensors);
    identity_kernel->SetDeviceInfo({kernel::GetFormatFromEnumToStr(x_tensor->format())},
                                   {output_kernel_tensor->GetStringFormat()}, {x_tensor->data_type()},
                                   {output_kernel_tensor->dtype_id()});

    identity_kernel->PackageInput(kIndex0, kernel::GetFormatFromEnumToStr(x_tensor->format()), &input_shape);
    identity_kernel->PackageOutput(kIndex0, output_shape);

    if (identity_kernel->Resize(input_kernel_tensors, output_kernel_tensors) != KRET_OK) {
      MS_LOG(EXCEPTION) << "Kernel identity resize failed";
    }
    auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());

    auto workspace_kernel_tensors =
      PyBoostUtils::CreateWorkSpaceKernelTensors(identity_kernel, device_context, "Identity");
    auto workspaces = PyBoostUtils::GetRawKernelTensor(workspace_kernel_tensors);
    if (!identity_kernel->Launch(input_kernel_tensors, workspaces, output_kernel_tensors, stream_ptr)) {
      MS_LOG(EXCEPTION) << "Launch kernel identity failed";
    }
    runtime::DeviceAddressUtils::ProcessCrossStreamAddress(prim::kPrimIdentity->name(), device_context, op->stream_id(),
                                                           input_kernel_tensors, output_kernel_tensors);
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
    auto input_shape = x_tensor->shape();
    auto output_shape = outputs[0]->shape();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    auto identity_kernel = std::make_shared<kernel::AclKernelMod>();
    const auto &input_x_address = x_tensor->device_address();
    const auto &output_address = outputs[0]->device_address();
    auto x_abs = x_tensor->ToAbstract()->Broaden();
    MS_EXCEPTION_IF_NULL(x_abs);
    auto input_kernel_tensor = std::make_shared<KernelTensor>(x_abs->GetShape(), x_abs->GetType(), nullptr);
    input_kernel_tensor->set_device_address(input_x_address);
    input_kernel_tensor->set_tensor_storage_info(x_tensor->storage_info());
    input_kernel_tensor->SetShapeVector(x_tensor->shape());
    input_kernel_tensor->set_format(x_tensor->format());
    if (!input_kernel_tensor->host_info_exist()) {
      input_kernel_tensor->SetHostInfo(std::make_shared<abstract::TensorShape>(x_tensor->shape()),
                                       std::make_shared<TensorType>(x_tensor->Dtype()), nullptr);
    }
    auto out_abs = outputs[0]->ToAbstract()->Broaden();
    MS_EXCEPTION_IF_NULL(out_abs);
    auto output_kernel_tensor = std::make_shared<KernelTensor>(out_abs->GetShape(), out_abs->GetType(), nullptr);
    output_kernel_tensor->set_device_address(output_address);
    output_kernel_tensor->set_tensor_storage_info(outputs[0]->storage_info());
    output_kernel_tensor->set_format(outputs[0]->format());
    output_kernel_tensor->SetShapeVector(outputs[0]->shape());
    if (!output_kernel_tensor->host_info_exist()) {
      output_kernel_tensor->SetHostInfo(std::make_shared<abstract::TensorShape>(outputs[0]->shape()),
                                        std::make_shared<TensorType>(outputs[0]->Dtype()), nullptr);
    }

    auto input_kernel_tensors = {input_kernel_tensor.get()};
    auto output_kernel_tensors = {output_kernel_tensor.get()};
    if (!std::static_pointer_cast<KernelMod>(identity_kernel)
           ->Init(prim::kPrimIdentity, input_kernel_tensors, output_kernel_tensors)) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Initialize acl kernel op[Identity] failed.";
    }
    identity_kernel->RefreshAclConverter(input_kernel_tensors);
    identity_kernel->SetDeviceInfo({kernel::GetFormatFromEnumToStr(x_tensor->format())},
                                   {output_kernel_tensor->GetStringFormat()}, {x_tensor->data_type()},
                                   {output_kernel_tensor->dtype_id()});

    identity_kernel->PackageInput(kIndex0, kernel::GetFormatFromEnumToStr(x_tensor->format()), &input_shape);
    identity_kernel->PackageOutput(kIndex0, output_shape);

    if (identity_kernel->Resize(input_kernel_tensors, output_kernel_tensors) != KRET_OK) {
      MS_LOG(EXCEPTION) << "Kernel identity resize failed";
    }
    auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());

    auto workspace_kernel_tensors =
      PyBoostUtils::CreateWorkSpaceKernelTensors(identity_kernel, device_context, "Identity");
    auto workspaces = PyBoostUtils::GetRawKernelTensor(workspace_kernel_tensors);
    if (!identity_kernel->Launch(input_kernel_tensors, workspaces, output_kernel_tensors, stream_ptr)) {
      MS_LOG(EXCEPTION) << "Launch kernel identity failed";
    }
    runtime::DeviceAddressUtils::ProcessCrossStreamAddress(prim::kPrimIdentity->name(), device_context, op->stream_id(),
                                                           input_kernel_tensors, output_kernel_tensors);
    MS_LOG(DEBUG) << "Run device task Identity end";
  }));
}

tensor::TensorPtr IdentityAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  OpRunner::InferOpOutput(op, x_tensor);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  IdentityCall(op, x_tensor);
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
