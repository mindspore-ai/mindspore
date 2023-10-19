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
#include "plugin/device/ascend/kernel/opapi/aclnn_functional_kernel_mod.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {

void AclnnFunctionalKernelMod::Init(const PrimitivePtr &prim, bool is_gradient_out) {
//  MS_EXCEPTION_IF_NULL(prim);
//  prim_ = prim;
//  kernel_name_ = prim->name();
  is_gradient_out_ = is_gradient_out;
  auto ms_context = MsContext::GetInstance();
  device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->Initialize();
}

void AclnnFunctionalKernelMod::CreateTensorAddress(const tensor::TensorPtr &tensor, const std::string &input_name,
                                                   bool is_gradient_out) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context_);

  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  if (device_address == nullptr) {
    auto tensor_size = LongToSize(tensor->data().nbytes());
    // Padding shape/format
    device_address = device_context_->device_res_manager_->CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT,
                                                                               tensor->data_type(), tensor->shape());

    device_address->set_from_persistent_mem(is_gradient_out || tensor->is_parameter());
    tensor->set_device_address(device_address);
  }

  if (device_address->GetPtr() != nullptr) {
    return;
  }

  if (!device_context_->device_res_manager_->AllocateMemory(device_address.get())) {
    MS_LOG(EXCEPTION) << "Allocate memory failed";
  }

  // Padding shape/format
  if (!device_address->SyncHostToDevice(tensor->shape(), LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                        tensor->data_c(), tensor->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
  }
}

device::DeviceAddressPtr AclnnFunctionalKernelMod::CreateWorkspaceAddress(const size_t &workspace_size) {
  MS_EXCEPTION_IF_NULL(device_context_);

  auto device_address =
    device_context_->device_res_manager_->CreateDeviceAddress(nullptr, workspace_size, "", kTypeUnknown, ShapeVector());
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetPtr() == nullptr &&
      !device_context_->device_res_manager_->AllocateMemory(device_address.get())) {
    MS_LOG(EXCEPTION) << "Allocate dynamic workspace memory failed";
  }

  return device_address;
}
}  // namespace kernel
}  // namespace mindspore
