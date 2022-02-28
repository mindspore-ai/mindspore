/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/environ/environ_gpu_destroy_all.h"
#include "kernel/environ_manager.h"

namespace mindspore {
namespace kernel {
bool EnvironDestroyAllGpuKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  // Check the output type.
  auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  if (output_type != TypeId::kNumberTypeBool) {
    MS_LOG(ERROR) << "The output type is invalid: " << output_type;
    return false;
  }

  InitSizeLists();
  return true;
}

void EnvironDestroyAllGpuKernelMod::InitSizeLists() { output_size_list_.push_back(sizeof(bool)); }

bool EnvironDestroyAllGpuKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &, void *) {
  MS_LOG(INFO) << "Clear the global environ data.";
  // Clear the global data which are generated in the kernel running.
  EnvironMgr::GetInstance().Clear();

  return true;
}
}  // namespace kernel
}  // namespace mindspore
