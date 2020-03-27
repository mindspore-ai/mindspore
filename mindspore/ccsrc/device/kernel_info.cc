/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "device/kernel_info.h"

namespace mindspore {
namespace device {
const kernel::KernelBuildInfo *KernelInfo::select_kernel_build_info() const { return select_kernel_build_info_.get(); }

kernel::KernelBuildInfoPtr KernelInfo::GetMutableSelectKernelBuildInfo() const { return select_kernel_build_info_; }

const DeviceAddress *KernelInfo::GetOutputAddr(size_t index) const {
  if (index >= output_address_list_.size()) {
    MS_LOG(ERROR) << "Index [" << index << "] out of range";
    return nullptr;
  }
  return output_address_list_[index].get();
}

DeviceAddressPtr KernelInfo::GetMutableOutputAddr(size_t index) const {
  if (index >= output_address_list_.size()) {
    MS_LOG(ERROR) << "Index [" << index << "] out of range";
    return nullptr;
  }
  return output_address_list_[index];
}

bool KernelInfo::OutputAddrExist(size_t index) const {
  if (index >= output_address_list_.size()) {
    return false;
  }
  return output_address_list_[index] != nullptr;
}

bool KernelInfo::SetOutputAddr(const DeviceAddressPtr &output_address, size_t index) {
  // parameter and valuenode
  if (kernel_mod_ == nullptr && index >= output_address_list_.size()) {
    for (size_t i = output_address_list_.size(); i <= index; i++) {
      output_address_list_.emplace_back(nullptr);
    }
  } else if (output_address_list_.empty()) {
    // set cnode
    for (size_t i = 0; i < kernel_mod_->GetOutputSizeList().size(); i++) {
      output_address_list_.emplace_back(nullptr);
    }
  }
  if (index >= output_address_list_.size()) {
    MS_LOG(ERROR) << "Index [" << index << "] out of range";
    return false;
  }
  output_address_list_[index] = output_address;
  return true;
}

DeviceAddress *KernelInfo::GetWorkspaceAddr(size_t index) const {
  if (index >= workspace_address_list_.size()) {
    MS_LOG(ERROR) << "Index [" << index << "] out of range";
    return nullptr;
  }
  return workspace_address_list_[index].get();
}

bool KernelInfo::SetWorkspaceAddr(const DeviceAddressPtr &output_address, size_t index) {
  if (workspace_address_list_.empty()) {
    // parameter and valuenode
    if (kernel_mod_ == nullptr) {
      workspace_address_list_.emplace_back(nullptr);
    } else {
      // set cnode
      for (size_t i = 0; i < kernel_mod_->GetWorkspaceSizeList().size(); i++) {
        workspace_address_list_.emplace_back(nullptr);
      }
    }
  }
  if (index >= workspace_address_list_.size()) {
    MS_LOG(ERROR) << "Index" << index << " out of range";
    return false;
  }
  workspace_address_list_[index] = output_address;
  return true;
}

void KernelInfo::set_kernel_mod(const kernel::KernelModPtr &kernel_mod) { kernel_mod_ = kernel_mod; }

kernel::KernelMod *KernelInfo::MutableKernelMod() const { return kernel_mod_.get(); }

const kernel::KernelMod *KernelInfo::kernel_mod() const { return kernel_mod_.get(); }

bool KernelInfo::operator==(const KernelInfo &other) const {
  if (stream_id_ != other.stream_id_ || stream_distinction_label_ != other.stream_distinction_label_ ||
      graph_id_ != other.graph_id_) {
    return false;
  }
  if ((select_kernel_build_info_ != nullptr && other.select_kernel_build_info_ == nullptr) ||
      (select_kernel_build_info_ == nullptr && other.select_kernel_build_info_ != nullptr)) {
    return false;
  }
  if (select_kernel_build_info_ != nullptr && other.select_kernel_build_info_ != nullptr) {
    if (!(*select_kernel_build_info_ == *(other.select_kernel_build_info_))) {
      return false;
    }
  }
  // Currently we only check whether both the kernel_mod_ are initialized or uninitialized.
  if ((kernel_mod_ == nullptr && other.kernel_mod_ != nullptr) ||
      (kernel_mod_ != nullptr && other.kernel_mod_ == nullptr)) {
    return false;
  }
  // Currently we only check whether both the sizes are equal of output_address_list_ and workspace_address_list_ or
  // not. We can complete this check in the future.
  if (output_address_list_.size() != other.output_address_list_.size() ||
      workspace_address_list_.size() != other.workspace_address_list_.size()) {
    return false;
  }
  return true;
}
}  // namespace device
}  // namespace mindspore
