/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/environ/environ_cpu_create.h"
#include "kernel/environ_manager.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
void EnvironCreateCpuKernelMod::InitKernel(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // Check the output handle.
  auto handle_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
  auto handle_shapes = AnfAlgo::GetOutputDeviceShape(node, 0);
  if (!EnvironMgr::GetInstance().IsScalarTensor(handle_type, handle_shapes)) {
    MS_LOG(EXCEPTION) << "The output handle checks invalid, kernel: " << node->fullname_with_scope();
  }

  handle_size_ = sizeof(int64_t);
  output_size_list_.push_back(handle_size_);
}

bool EnvironCreateCpuKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                       const std::vector<AddressPtr> &outputs) {
  // Generate an unique handle.
  int64_t env_handle = EnvironMgr::GetInstance().Create();

  auto output = GetDeviceAddress<int64_t>(outputs, 0);
  output[0] = env_handle;
  MS_LOG(DEBUG) << "Create env handle: " << output[0];

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EnvironCreate, EnvironCreateCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
