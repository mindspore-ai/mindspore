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

#include "backend/kernel_compiler/gpu/rl/tensor_array_clear_kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/gpu/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::TensorArrayPtr;
TensorArrayClearKernel::TensorArrayClearKernel() {}

const std::vector<size_t> &TensorArrayClearKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &TensorArrayClearKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &TensorArrayClearKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool TensorArrayClearKernel::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  InitSizeLists();
  return true;
}

void TensorArrayClearKernel::InitSizeLists() {
  input_size_list_.push_back(sizeof(int64_t));
  output_size_list_.push_back(sizeof(int64_t));
}

bool TensorArrayClearKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<AddressPtr> &, void *stream) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  MS_ERROR_IF_NULL(handle_addr);
  int64_t handle = 0;
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(&handle, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost,
                                             reinterpret_cast<cudaStream_t>(stream)),
                             "Get handle to host failed");
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle);
  MS_ERROR_IF_NULL(tensors_);
  // Clear TensorArray valid size, but keep the memory.
  tensors_->Clear();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
