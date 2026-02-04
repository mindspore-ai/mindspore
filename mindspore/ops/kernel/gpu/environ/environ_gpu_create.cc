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

#include "kernel/gpu/environ/environ_gpu_create.h"
#include "include/runtime/hardware_abstract/kernel_base/environ_manager.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
int EnvironCreateGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  // Check the output handle.
  auto handle_type = outputs[kIndex0]->dtype_id();
  const auto &handle_shapes = outputs[kIndex0]->GetShapeVector();
  if (!EnvironMgr::GetInstance().IsScalarTensor(handle_type, handle_shapes)) {
    MS_LOG(ERROR) << "The output handle checks invalid, kernel: " << kernel_name_;
    return false;
  }
  handle_size_ = sizeof(int64_t);

  output_size_list_.clear();
  output_size_list_.push_back(handle_size_);
  return KRET_OK;
}

bool EnvironCreateGpuKernelMod::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto output = GetDeviceAddress<int64_t>(outputs, kIndex0);

  // Generate an unique handle.
  int64_t env_handle = EnvironMgr::GetInstance().Create();
  MS_LOG(DEBUG) << "Create env handle: " << env_handle;

  // Copy handle to output.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output, &env_handle, handle_size_, cudaMemcpyHostToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "Copy env handle failed.");
  return true;
}
}  // namespace kernel
}  // namespace mindspore
