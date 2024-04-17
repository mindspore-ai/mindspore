/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include "runtime/device/launch_kernel.h"
#include "include/backend/anf_runtime_algorithm.h"
namespace mindspore::device {
std::vector<kernel::KernelTensor *> LaunchKernel::ObtainKernelAddress(const std::vector<size_t> &list,
                                                                      std::vector<uint8_t *> *addr) {
  MS_EXCEPTION_IF_NULL(addr);
  std::vector<kernel::KernelTensor *> kernel_tensors;
  if (addr->size() < list.size()) {
    MS_LOG_EXCEPTION << "Error addr size!";
  }
  for (size_t i = 0; i < list.size(); ++i) {
    auto size = AlignSizeForLaunchKernel(list[i]);
    (*addr)[i] = AllocDeviceMem(size);
    MS_EXCEPTION_IF_NULL((*addr)[i]);
    auto kernel_tensor = std::make_shared<kernel::KernelTensor>();
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    kernel_tensor->set_device_ptr((*addr)[i]);
    kernel_tensor->set_size(size);
    kernel_tensors.push_back(kernel_tensor.get());
  }
  return kernel_tensors;
}

std::vector<kernel::KernelTensor *> LaunchKernel::ObtainKernelInputs(const std::vector<size_t> &inputs_list,
                                                                     const std::vector<uint8_t *> &inputs_addr) {
  std::vector<kernel::KernelTensor *> kernel_inputs;
  if (inputs_list.size() != inputs_addr.size()) {
    MS_LOG(ERROR) << "input_list size should equal to input_addr_ size, input_list size: " << inputs_list.size()
                  << ", input_addr_ size: " << inputs_addr.size();
  }
  for (size_t i = 0; i < inputs_list.size(); ++i) {
    auto input_size = AlignSizeForLaunchKernel(inputs_list[i]);
    auto input = std::make_shared<kernel::KernelTensor>();
    MS_EXCEPTION_IF_NULL(input);
    auto addr = inputs_addr[i];
    MS_EXCEPTION_IF_NULL(addr);
    input->set_device_ptr(addr);
    input->set_size(input_size);
    kernel_inputs.push_back(input.get());
  }
  return kernel_inputs;
}

std::vector<kernel::KernelTensor *> LaunchKernel::ObtainKernelOutputs(const std::vector<size_t> &outputs_list) {
  // init output_addr_
  outputs_addr_ = std::vector<uint8_t *>(outputs_list.size(), nullptr);
  auto kernel_outputs = ObtainKernelAddress(outputs_list, &outputs_addr_);
  return kernel_outputs;
}

std::vector<kernel::KernelTensor *> LaunchKernel::ObtainKernelWorkspaces(const std::vector<size_t> &workspaces_list) {
  std::vector<kernel::KernelTensor *> kernel_workspace;
  if (workspaces_list.empty()) {
    return kernel_workspace;
  }
  // init workspace_addr_
  workspaces_addr_ = std::vector<uint8_t *>(workspaces_list.size(), nullptr);
  kernel_workspace = ObtainKernelAddress(workspaces_list, &workspaces_addr_);
  return kernel_workspace;
}

void LaunchKernel::LaunchSingleKernel(const AnfNodePtr &node, const std::vector<uint8_t *> &inputs_addr) {
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  // obtain kernel inputs
  auto inputs = session::AnfRuntimeAlgorithm::GetOrCreateAllInputKernelTensors(node);
  std::vector<size_t> input_size_list;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_size_list),
                 [](auto input) { return input->size(); });
  auto kernel_inputs = ObtainKernelInputs(input_size_list, inputs_addr);
  // obtain kernel outputs
  auto kernel_outputs = ObtainKernelOutputs(kernel_mod_->GetOutputSizeList());
  // obtain kernel workspace
  auto kernel_workspaces = ObtainKernelWorkspaces(kernel_mod_->GetWorkspaceSizeList());
  // launch
  auto ret_status = kernel_mod_->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_);
  if (!ret_status) {
    MS_LOG(ERROR) << "Launch single kernel failed.";
  }
}

void LaunchKernel::FreeOutputAndWorkspaceDeviceMem() {
  // free outputs_addr and workspaces_addr_
  for (size_t i = 0; i < outputs_addr_.size(); ++i) {
    if (outputs_addr_[i] != nullptr) {
      FreeDeviceMem(outputs_addr_[i]);
      outputs_addr_[i] = nullptr;
    }
  }
  for (size_t i = 0; i < workspaces_addr_.size(); ++i) {
    if (workspaces_addr_[i] != nullptr) {
      FreeDeviceMem(workspaces_addr_[i]);
      workspaces_addr_[i] = nullptr;
    }
  }
  outputs_addr_.clear();
  workspaces_addr_.clear();
}
}  // namespace mindspore::device
