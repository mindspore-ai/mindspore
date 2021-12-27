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

#include "backend/kernel_compiler/gpu/environ/environ_gpu_create.h"
#include "backend/kernel_compiler/environ_manager.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
const std::vector<size_t> &EnvironCreateGpuKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &EnvironCreateGpuKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &EnvironCreateGpuKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool EnvironCreateGpuKernel::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  // Check the output handle.
  auto handle_type = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  auto handle_shapes = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (!EnvironMgr::GetInstance().IsScalarTensor(handle_type, handle_shapes)) {
    MS_LOG(ERROR) << "The output handle checks invalid, kernel: " << kernel_node->fullname_with_scope();
    return false;
  }
  handle_size_ = sizeof(int64_t);

  InitSizeLists();
  return true;
}

void EnvironCreateGpuKernel::InitSizeLists() { output_size_list_.push_back(handle_size_); }

bool EnvironCreateGpuKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto output = GetDeviceAddress<int64_t>(outputs, 0);

  // Generate an unique handle.
  int64_t env_handle = EnvironMgr::GetInstance().Create();
  MS_LOG(DEBUG) << "Create env handle: " << env_handle;

  // Copy handle to output.
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(output, &env_handle, handle_size_, cudaMemcpyHostToDevice,
                                             reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Copy env handle failed.");
  return true;
}
}  // namespace kernel
}  // namespace mindspore
