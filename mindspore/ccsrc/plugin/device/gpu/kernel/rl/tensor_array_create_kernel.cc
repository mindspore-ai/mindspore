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
#include "plugin/device/gpu/kernel/rl/tensor_array_create_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::gpu::GPUTensorArray;
using mindspore::device::gpu::GPUTensorArrayPtr;
TensorArrayCreateKernelMod::TensorArrayCreateKernelMod() : is_dynamic_(true), size_(0), type_(nullptr) {}

bool TensorArrayCreateKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  shapes_ = GetAttr<std::vector<int64_t>>(kernel_node, "element_shape");

  type_ = GetAttr<TypePtr>(kernel_node, "dtype");
  size_ = GetAttr<int64_t>(kernel_node, "size");
  is_dynamic_ = GetAttr<bool>(kernel_node, "dynamic_size");
  name_ = GetAttr<std::string>(kernel_node, "name");
  InitSizeLists();
  return true;
}

void TensorArrayCreateKernelMod::InitSizeLists() { output_size_list_.push_back(sizeof(int64_t)); }

bool TensorArrayCreateKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // Create a tensorarray, and generate an unique handle.
  int64_t tensor_array_handle = TensorArrayMgr::GetInstance().GetHandleCount();
  auto name = "GPUTensorArray_" + name_ + "_" + std::to_string(tensor_array_handle);
  GPUTensorArrayPtr tensor_array = std::make_shared<GPUTensorArray>(name, type_, shapes_);
  MS_EXCEPTION_IF_NULL(tensor_array);
  tensor_array->SetMaxSize(size_, is_dynamic_);
  auto out_addr = GetDeviceAddress<int64_t>(outputs, 0);
  // Set handle to out_addr.
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(out_addr, &tensor_array_handle, sizeof(int64_t), cudaMemcpyHostToDevice,
                                             reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Create TensorArray failed");
  MS_LOG(DEBUG) << "Create handle id " << tensor_array_handle;
  // Put tensorarray to a saved map : map<handle, tensorarray> in tensorarray manager.
  // And increase the handle count automatically in AddTensorArray function.
  TensorArrayMgr::GetInstance().AddTensorArray(tensor_array_handle, tensor_array);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
