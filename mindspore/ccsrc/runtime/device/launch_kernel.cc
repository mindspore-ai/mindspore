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

#include "runtime/device/launch_kernel.h"

#include <vector>
#include <memory>

namespace mindspore::device {
std::vector<kernel::AddressPtr> LaunchKernel::ObtainKernelAddress(const std::vector<size_t> &list,
                                                                  std::vector<uint8_t *> *addr) {
  std::vector<kernel::AddressPtr> kernel_address;
  for (size_t i = 0; i < list.size(); ++i) {
    auto size = AlignSizeForLaunchKernel(list[i]);
    (*addr)[i] = AllocDeviceMem(size);
    auto address = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(address);
    address->addr = (*addr)[i];
    MS_EXCEPTION_IF_NULL(address->addr);
    address->size = size;
    kernel_address.push_back(address);
  }
  return kernel_address;
}

std::vector<kernel::AddressPtr> LaunchKernel::ObtainKernelInputs(const std::vector<size_t> &inputs_list,
                                                                 const std::vector<uint8_t *> &inputs_addr) {
  std::vector<kernel::AddressPtr> kernel_inputs;
  if (inputs_list.size() != inputs_addr.size()) {
    MS_LOG(ERROR) << "input_list size should equal to input_addr_ size";
  }
  for (size_t i = 0; i < inputs_list.size(); ++i) {
    auto input_size = AlignSizeForLaunchKernel(inputs_list[i]);
    auto input = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(input);
    input->addr = inputs_addr[i];
    MS_EXCEPTION_IF_NULL(input->addr);
    input->size = input_size;
    kernel_inputs.push_back(input);
  }
  return kernel_inputs;
}

std::vector<kernel::AddressPtr> LaunchKernel::ObtainKernelOutputs(const std::vector<size_t> &outputs_list) {
  // init output_addr_
  outputs_addr_ = std::vector<uint8_t *>(outputs_list.size(), nullptr);
  auto kernel_outputs = ObtainKernelAddress(outputs_list, &outputs_addr_);
  return kernel_outputs;
}

std::vector<kernel::AddressPtr> LaunchKernel::ObtainKernelWorkspaces(const std::vector<size_t> &workspaces_list) {
  std::vector<kernel::AddressPtr> kernel_workspace;
  if (workspaces_list.empty()) {
    return kernel_workspace;
  }
  // init workspace_addr_
  workspaces_addr_ = std::vector<uint8_t *>(workspaces_list.size(), nullptr);
  kernel_workspace = ObtainKernelAddress(workspaces_list, &workspaces_addr_);
  return kernel_workspace;
}

void LaunchKernel::LaunchSingleKernel(const std::vector<uint8_t *> &inputs_addr) {
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  // obtain kernel inputs
  auto kernel_inputs = ObtainKernelInputs(kernel_mod_->GetInputSizeList(), inputs_addr);
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
