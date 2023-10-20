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

#include "plugin/device/cpu/kernel/pyboost/mul_cpu.h"
#include "arithmetic_cpu_kernel.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "ops/mul.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MulCPU::Call(const tensor::TensorPtr &x, const tensor::TensorPtr &y) {
  Infer(primitive_, x, y);
  auto kernel = std::make_shared<ArithmeticCpuKernelMod>("Mul");
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kCPUDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, x, "x");
  runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, y, "y");
  runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, outputs_[0], "out", false);
  auto input_x = TensorToKernelTensor(x, device_context);
  auto input_y = TensorToKernelTensor(y, device_context);
  auto output = TensorToKernelTensor(outputs_[0], device_context);
  auto base_op = std::make_shared<ops::Mul>("Mul");
  kernel->Init(base_op, {input_x, input_y}, {output});
  kernel->Resize(base_op, {input_x, input_y}, {output}, {});
  auto workspace_sizes = kernel->GetWorkspaceSizeList();
  kernel::AddressPtrList workspaces;
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto workspace_device_address =
      runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context, workspace_sizes[i]);
    (void)workspaces.emplace_back(std::make_shared<kernel::Address>(workspace_device_address->GetMutablePtr(),
                                                                    workspace_device_address->GetSize()));
  }
  vector<AddressPtr> inputs;
  vector<AddressPtr> outputs;
  (void)inputs.emplace_back(input_x->GetData());
  (void)inputs.emplace_back(input_y->GetData());
  (void)outputs.emplace_back(output->GetData());
  kernel->Launch(inputs, {workspaces}, outputs);
  return outputs_[0];
}

tensor::TensorPtr MulCPU::Call(const tensor::TensorPtr &x, const ScalarPtr &y) {
  auto x_type = x->data_type();
  auto tensor_y = pyboost::ScalarToTensor(y, TypeIdToType(x_type));
  Infer(primitive_, x, tensor_y);
  auto kernel = std::make_shared<ArithmeticCpuKernelMod>("Mul");
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kCPUDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, x, "x");
  runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, tensor_y, "y");
  runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, outputs_[0], "out", false);
  auto input_x = TensorToKernelTensor(x, device_context);
  auto input_y = TensorToKernelTensor(tensor_y, device_context);
  auto output = TensorToKernelTensor(outputs_[0], device_context);
  auto base_op = std::make_shared<ops::Mul>("Mul");
  kernel->Init(base_op, {input_x, input_y}, {output});
  kernel->Resize(base_op, {input_x, input_y}, {output}, {});
  auto workspace_sizes = kernel->GetWorkspaceSizeList();
  kernel::AddressPtrList workspaces;
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto workspace_device_address =
      runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context, workspace_sizes[i]);
    (void)workspaces.emplace_back(std::make_shared<kernel::Address>(workspace_device_address->GetMutablePtr(),
                                                                    workspace_device_address->GetSize()));
  }
  vector<AddressPtr> inputs;
  vector<AddressPtr> outputs;
  (void)inputs.emplace_back(input_x->GetData());
  (void)inputs.emplace_back(input_y->GetData());
  (void)outputs.emplace_back(output->GetData());
  kernel->Launch(inputs, {workspaces}, outputs);
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
