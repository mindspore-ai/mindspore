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

#include "plugin/device/gpu/kernel/pyboost/add_ext_gpu.h"
#include "mindspore/ccsrc/plugin/device/gpu/kernel/math/binary_ops_gpu_kernel.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "ops/add.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AddExtGPU::Call(const tensor::TensorPtr &self, const tensor::TensorPtr &other,
                                  const ScalarPtr &alpha) {
  // TODO: (CARRY) dyn_shape_dev
  InferOutput(self, other);

  auto device_context = PyBoostUtils::GetDeviceContext(kGPUDevice);

  Contiguous(self);
  Contiguous(other);
  runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, self, "self");
  runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, other, "other");
  runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, outputs_[0], "out");
  auto input_x = TensorToKernelTensor(self, device_context);
  auto input_y = TensorToKernelTensor(other, device_context);
  auto output = TensorToKernelTensor(outputs_[0], device_context);
  auto base_op = std::make_shared<ops::Add>("Add");
  auto kernel = std::make_shared<BroadcastOptGpuKernelMod>("Add");
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
  auto stream_ptr = device::gpu::GPUDeviceManager::GetInstance().GetStream(kDefaultStreamIndex);
  kernel->Launch(inputs, {workspaces}, outputs, stream_ptr);
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
