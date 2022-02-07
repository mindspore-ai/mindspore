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

#include "backend/kernel_compiler/gpu/environ/environ_gpu_destroy_all.h"
#include "backend/kernel_compiler/environ_manager.h"

namespace mindspore {
namespace kernel {
const std::vector<size_t> &EnvironDestroyAllGpuKernelMod::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &EnvironDestroyAllGpuKernelMod::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &EnvironDestroyAllGpuKernelMod::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool EnvironDestroyAllGpuKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
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
