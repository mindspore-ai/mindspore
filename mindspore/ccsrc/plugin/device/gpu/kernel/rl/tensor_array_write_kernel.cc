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
#include <memory>
#include "plugin/device/gpu/kernel/rl/tensor_array_write_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSecondInputIndex = 2;
using mindspore::device::TensorArrayMgr;
using mindspore::device::gpu::GPUTensorArray;
using mindspore::device::gpu::GPUTensorArrayPtr;
TensorArrayWriteKernelMod::TensorArrayWriteKernelMod() : value_size_(0) {}

bool TensorArrayWriteKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kSecondInputIndex);
  shapes_ = AnfAlgo::GetInputDeviceShape(kernel_node, kSecondInputIndex);
  value_size_ = GetTypeByte(TypeIdToType(type_)) * SizeOf(shapes_);

  InitSizeLists();
  return true;
}

void TensorArrayWriteKernelMod::InitSizeLists() {
  input_size_list_.push_back(sizeof(int64_t));
  input_size_list_.push_back(sizeof(int64_t));
  input_size_list_.push_back(value_size_);
  output_size_list_.push_back(sizeof(int64_t));
}

bool TensorArrayWriteKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<AddressPtr> &, void *stream) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto index = GetDeviceAddress<int64_t>(inputs, 1);
  auto value = GetDeviceAddress<unsigned char>(inputs, kSecondInputIndex);
  MS_ERROR_IF_NULL(handle_addr);
  MS_ERROR_IF_NULL(index);
  MS_ERROR_IF_NULL(value);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  MS_ERROR_IF_NULL(cuda_stream);
  int64_t index_host = 0;
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(&index_host, index, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
                             "Get indexd failed");
  int64_t handle = 0;
  CHECK_CUDA_RET_WITH_EXCEPT(
    kernel_node_, cudaMemcpyAsync(&handle, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
    "For 'TensorArrayWrite', Get handle to host failed");
  if (cudaStreamQuery(cuda_stream) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "cuda Stream Sync Failed");
  }
  GPUTensorArrayPtr tensors_ =
    std::dynamic_pointer_cast<GPUTensorArray>(TensorArrayMgr::GetInstance().GetTensorArray(handle));
  MS_EXCEPTION_IF_NULL(tensors_);
  if (!tensors_->CheckValue(type_, shapes_)) {
    MS_LOG(EXCEPTION) << "Invalid input data for tensor array write op.";
  }
  // Manage the value : create/reuse a device memory, and copy the input value to it.
  AddressPtr dev_addr = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(dev_addr);
  if (tensors_->GetRealSize() > LongToSize(index_host)) {
    dev_addr->addr = tensors_->Read(index_host)->addr;
  } else {
    dev_addr->addr = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(value_size_);
    MS_LOG(DEBUG) << "Create tensor " << dev_addr->addr << ", size " << value_size_;
  }
  MS_EXCEPTION_IF_NULL(dev_addr->addr);
  dev_addr->size = value_size_;
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(dev_addr->addr, value, value_size_, cudaMemcpyDeviceToDevice, cuda_stream),
                             "Copy value failed");

  if (tensors_->Write(index_host, dev_addr)) {
    MS_LOG(DEBUG) << "Write to tensorarry succeed, index " << index_host;
  } else {
    MS_LOG(EXCEPTION) << "Failed to write.";
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
