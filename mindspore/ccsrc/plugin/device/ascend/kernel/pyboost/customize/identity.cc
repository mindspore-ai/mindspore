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
tensor::TensorPtr IdentityAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  OpRunner::InferOpOutput(op, x_tensor);

  PyBoostUtils::PrepareOpInputs(op->device_context(), x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, x_tensor]() {
    MS_LOG(DEBUG) << "Run device task Identity start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
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
      output_address->kernel_tensor()->SetHostInfo(std::make_shared<abstract::TensorShape>(outputs[0]->shape()),
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
    auto input_shape = x_tensor->shape();

    identity_kernel->PackageInput(kIndex0, input_x_address->format(), &input_shape);
    identity_kernel->PackageOutput(kIndex0, outputs[0]->shape());
    identity_kernel->SetNeedConvertHostTensor(true);

    if (identity_kernel->Resize(input_kernel_tensors, output_kernel_tensors) != KRET_OK) {
      MS_LOG(EXCEPTION) << "Kernel identity resize failed";
    }
    auto stream_ptr = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);

    auto workspace_sizes = identity_kernel->GetWorkspaceSizeList();
    std::vector<kernel::KernelTensor *> workspaces;
    workspaces.reserve(workspace_sizes.size());
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      auto kernel_tensor = std::make_shared<KernelTensor>(
        nullptr, workspace_sizes[i], Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
        device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
      auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
      MS_EXCEPTION_IF_NULL(device_address);
      if (device_address->GetPtr() == nullptr &&
          !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
        MS_LOG(EXCEPTION) << "Allocate dynamic workspace memory failed";
      }
      (void)workspaces.emplace_back(device_address->kernel_tensor().get());
      MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->device_ptr()
                    << " size:" << workspaces.back()->size();
    }

    if (!identity_kernel->Launch(input_kernel_tensors, workspaces, output_kernel_tensors, stream_ptr)) {
      MS_LOG(EXCEPTION) << "Launch kernel identity failed";
    }
    MS_LOG(DEBUG) << "Run device task Identity end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
