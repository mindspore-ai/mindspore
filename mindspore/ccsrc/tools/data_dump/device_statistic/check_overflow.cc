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
#include "tools/data_dump/device_statistic/check_overflow.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "include/utils/common.h"
#include "tools/data_dump/debugger/debugger_utils.h"
#include "tools/data_dump/device_statistic/kernel_factory.h"
#include "tools/data_dump/device_statistic/mem_manager.h"

namespace mindspore {
namespace datadump {

std::vector<KernelTensor *> CheckOverflowKernel::CheckInputs(std::vector<KernelTensor *> inputs) {
  std::vector<KernelTensor *> check_kernel_tensors;
  static std::set<TypeId> warning_once;

  for (size_t i = 0; i < inputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    auto type = inputs[i]->dtype_id();
    if (supported_dtype_.find(type) != supported_dtype_.end()) {
      check_kernel_tensors.emplace_back(inputs[i]);
    } else {
      if (warning_once.find(type) != warning_once.end()) {
        break;
      } else {
        warning_once.insert(type);
        MS_LOG(WARNING) << "Overflow detection does not support " << TypeIdToType(type) << " !";
      }
    }
  }
  return check_kernel_tensors;
}

std::vector<KernelTensorPtr> CheckOverflowKernel::GetWorkSpaceDeviceAddressList(
  const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = kernel_mod_->Resize(inputs, outputs);
  if (ret) {
    MS_LOG(EXCEPTION) << "Call Resize error, error id is " << ret;
  }
  auto work_space = kernel_mod_->GetWorkspaceSizeList();
  std::vector<KernelTensorPtr> workspace_list(inputs.size(), nullptr);
  for (size_t i = 0; i < work_space.size(); i++) {
    if (!work_space.empty() && work_space[i] != 0) {
      MS_VLOG(VL_DUMP) << "Statistic kernel name is " << kernel_name_ << ", workspace size is " << work_space[i]
                       << "input index is " << i << ", input shape is " << inputs[i]->GetShapeVector() << ", dtype is "
                       << TypeIdToString(inputs[i]->dtype_id()) << " , StreamId is " << stream_id_;
      MS_EXCEPTION_IF_NULL(device_context_);
      workspace_list[i] = DumpMemManager::GetInstance().GetWorkSpaceTensor(device_context_, stream_id_, work_space[i]);
    }
  }
  return workspace_list;
}

KernelTensorPtr CheckOverflowKernel::LaunchKernelAsync(std::vector<KernelTensor *> inputs,
                                                       const std::uint32_t stream_id) {
  stream_id_ = stream_id;
  std::vector<KernelTensor *> selected_inputs = CheckInputs(inputs);
  if (selected_inputs.empty()) {
    return nullptr;
  }

  // Output memory reuse
  auto output_kernel_tensor = GetOutputDeviceAddress(kNumberTypeBool);
  std::vector<KernelTensor *> outputs{output_kernel_tensor.get()};

  // Get workspace
  auto workspace_list = GetWorkSpaceDeviceAddressList(selected_inputs, outputs);
  std::vector<KernelTensor *> workspaces;
  if (!workspace_list.empty()) {
    for (size_t i = 0; i < workspace_list.size(); i++) {
      workspaces.emplace_back(workspace_list[i].get());
    }
  }

  MS_VLOG(VL_DUMP) << "The workspaces size is" << workspaces.size();

  MS_EXCEPTION_IF_NULL(kernel_mod_);

  void *stream_ptr = device_context_->device_res_manager_->GetStream(stream_id_);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  bool ret = kernel_mod_->Launch(selected_inputs, workspaces, outputs, stream_ptr);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Device cal overflow check, launch " << kernel_name_ << "error";
  }
  return output_kernel_tensor;
}

REGISTER_KERNEL(KCheckOverflow, CheckOverflowKernel);

}  // namespace datadump
}  // namespace mindspore
