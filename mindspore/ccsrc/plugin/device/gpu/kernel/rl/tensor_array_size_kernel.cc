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

#include "plugin/device/gpu/kernel/rl/tensor_array_size_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
TensorArraySizeKernelMod::TensorArraySizeKernelMod() {}

bool TensorArraySizeKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  return true;
}

int TensorArraySizeKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  output_size_list_.push_back(sizeof(int64_t));
  return KRET_OK;
}

bool TensorArraySizeKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto out_addr = GetDeviceAddress<int64_t>(outputs, 0);
  MS_ERROR_IF_NULL(out_addr);
  MS_ERROR_IF_NULL(handle_addr);
  int64_t handle = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&handle, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "For 'TensorArraySize', get handle to host failed");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "For 'TensorArraySize', cudaStreamSyncFailed");
  }
  auto tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle);
  MS_ERROR_IF_NULL(tensors_);
  int64_t valid_size = SizeToLong(tensors_->GetValidSize());
  MS_LOG(DEBUG) << "Launch TensorArraySize, valid size is " << valid_size;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(out_addr, &valid_size, sizeof(int64_t), cudaMemcpyHostToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "Get valid size failed");

  return true;
}
}  // namespace kernel
}  // namespace mindspore
