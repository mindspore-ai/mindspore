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
#include "tools/data_dump/device_statistic/statistic_kernel.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/utils/common.h"
#include "ir/tensor_new.h"
#include "tools/data_dump/debugger/debugger_utils.h"
#include "tools/data_dump/device_statistic/mem_manager.h"

namespace mindspore {

namespace datadump {

TensorPtr SyncDeviceToHostTensor(KernelTensorPtr kernel_tensor) {
  if (!kernel_tensor) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_addr = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  auto dtype_id = kernel_tensor->dtype_id();
  const auto &shape_vec = kernel_tensor->GetShapeVector();

  mindspore::tensor::TensorPtr out_tensor = tensor::from_spec(dtype_id, shape_vec, device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(out_tensor->device_address());

  device::DeviceContextKey host_key = {device_addr->GetDeviceType(), device_addr->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  if (!host_context->device_res_manager_->SyncAllStreams() ||
      !SyncCopy(out_tensor, kernel_tensor.get(), device_addr->stream_id())) {
    const auto &dst_address = dynamic_cast<device::DeviceAddress *>(out_tensor->device_address().get());
    MS_EXCEPTION_IF_NULL(dst_address);
    MS_LOG(EXCEPTION) << "Convert format or Copy device mem to host failed, from device address:"
                      << device_addr->ToString() << " to:" << dst_address->ToString();
  }
  return out_tensor;
}

KernelTensorPtr StatisticKernel::GetWorkSpaceDeviceAddress(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  auto ret = kernel_mod_->Resize(inputs, outputs);
  if (ret) {
    MS_LOG(EXCEPTION) << "Call Resize error, error id is " << ret;
  }
  auto work_space = kernel_mod_->GetWorkspaceSizeList();
  if (!work_space.empty() && work_space[0] != 0) {
    MS_VLOG(VL_DUMP) << "Statistic kernel name is " << kernel_name_ << ", workspace size is " << work_space[0]
                     << ", input shape is " << inputs[0]->GetShapeVector() << ", dtype is "
                     << TypeIdToString(inputs[0]->dtype_id());
    return DumpMemManager::GetInstance().GetWorkSpaceTensor(device_context_, stream_id_, work_space[0]);
  }
  return nullptr;
}

KernelTensorPtr StatisticKernel::GetOutputDeviceAddress(TypeId dtype_id) {
  return DumpMemManager::GetInstance().GetOutputTensor(device_context_, stream_id_, dtype_id);
}

std::vector<KernelTensorPtr> StatisticKernel::GetExtraInputsDeviceAddress(KernelTensor *) {
  return std::vector<KernelTensorPtr>();
}

std::vector<KernelTensorPtr> StatisticKernel::LaunchKernelAsync(KernelTensor *input, const uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(input);
  stream_id_ = stream_id;
  std::vector<KernelTensor *> inputs{input};
  auto extra_inputs = GetExtraInputsDeviceAddress(input);
  std::vector<KernelTensorPtr> res;
  std::transform(extra_inputs.begin(), extra_inputs.end(), std::back_inserter(inputs),
                 [](const auto &extra_input) { return extra_input.get(); });
  auto output_kernel_tensor = GetOutputDeviceAddress(input->dtype_id());
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  std::vector<KernelTensor *> outputs{output_kernel_tensor.get()};

  MS_EXCEPTION_IF_NULL(kernel_mod_);

  void *stream_ptr = device_context_->device_res_manager_->GetStream(stream_id_);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto workspace_kernel_tensor = GetWorkSpaceDeviceAddress(inputs, outputs);
  res.emplace_back(output_kernel_tensor);
  std::vector<KernelTensor *> workspace;
  if (workspace_kernel_tensor) {
    workspace.emplace_back(workspace_kernel_tensor.get());
  }
  MS_VLOG(VL_DUMP) << "Start launch statistic kernel, kernel name is " << kernel_name_ << ", stream id is "
                   << stream_id_;
  bool ret = kernel_mod_->Launch(inputs, workspace, outputs, stream_ptr);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Device cal statistic, launch " << kernel_name_ << "error";
  }
  return res;
}

}  // namespace datadump
}  // namespace mindspore
