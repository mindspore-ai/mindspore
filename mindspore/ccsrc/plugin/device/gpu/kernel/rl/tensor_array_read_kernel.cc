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

#include "plugin/device/gpu/kernel/rl/tensor_array_read_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::TensorArrayPtr;
TensorArrayReadKernelMod::TensorArrayReadKernelMod() : value_size_(0), type_(nullptr) {}

bool TensorArrayReadKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  return true;
}

int TensorArrayReadKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  shapes_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr("element_shape"));
  type_ = GetValue<TypePtr>(primitive_->GetAttr("dtype"));
  value_size_ = GetTypeByte(type_);
  for (auto i : shapes_) {
    value_size_ *= i;
  }
  output_size_list_.clear();
  output_size_list_.push_back(value_size_);
  return KRET_OK;
}

bool TensorArrayReadKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                      const std::vector<KernelTensor *> &outputs, void *stream) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  MS_ERROR_IF_NULL(handle_addr);
  auto index = GetDeviceAddress<int64_t>(inputs, 1);
  MS_ERROR_IF_NULL(index);
  auto out_value = GetDeviceAddress<unsigned char>(outputs, 0);
  MS_ERROR_IF_NULL(out_value);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  MS_ERROR_IF_NULL(cuda_stream);
  int64_t index_host = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&index_host, index, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
    "For 'TensorArrayRead', get index to host failed");
  int64_t handle = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&handle, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
    "For 'TensorArrayRead', get handle to host failed");
  if (cudaStreamQuery(cuda_stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "cuda Stream Sync Failed");
  }
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle);
  MS_ERROR_IF_NULL(tensors_);
  if (!tensors_->CheckReadIndexLogical(index_host)) {
    MS_LOG(EXCEPTION) << "Invalid index " << index_host << " for read.";
  }
  auto value_addr = tensors_->Read(index_host);
  MS_LOG(DEBUG) << "Read value index:" << index_host;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(out_value, value_addr->addr, value_size_, cudaMemcpyDeviceToDevice, cuda_stream),
    "Get value failed");
  return true;
}
}  // namespace kernel
}  // namespace mindspore
