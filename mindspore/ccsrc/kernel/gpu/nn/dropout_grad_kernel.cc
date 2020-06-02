/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/nn/dropout_grad_kernel.h"
#include "kernel/gpu/cuda_impl/dropout_impl.cuh"

namespace mindspore {
namespace kernel {
DropoutGradGpuFwdKernel::DropoutGradGpuFwdKernel()
    : cudnn_handle_(nullptr), is_null_input_(false), num_count_(0), keep_prob_(0.0) {}

DropoutGradGpuFwdKernel::~DropoutGradGpuFwdKernel() { DestroyResource(); }

const std::vector<size_t> &DropoutGradGpuFwdKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &DropoutGradGpuFwdKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &DropoutGradGpuFwdKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool DropoutGradGpuFwdKernel::Init(const CNodePtr &kernel_node) {
  InitResource();

  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(ERROR) << "Argument number is " << input_num << ", but DropoutGradGpuFwdKernel needs 2.";
    return false;
  }

  auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  is_null_input_ = CHECK_NULL_INPUT(input_shape);
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }

  num_count_ = 1;
  for (size_t x : input_shape) {
    num_count_ *= x;
  }
  keep_prob_ = GetValue<float>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("keep_prob"));

  InitSizeLists();
  return true;
}

void DropoutGradGpuFwdKernel::InitResource() {
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
}

void DropoutGradGpuFwdKernel::DestroyResource() noexcept {}

void DropoutGradGpuFwdKernel::InitSizeLists() {
  size_t dy_size = num_count_ * sizeof(float);
  size_t mask_size = dy_size;
  size_t dx_size = dy_size;
  size_t workspace_size = 0;

  input_size_list_.push_back(dy_size);
  input_size_list_.push_back(mask_size);
  output_size_list_.push_back(dx_size);
  workspace_size_list_.push_back(workspace_size);
}

bool DropoutGradGpuFwdKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }

  auto *dy = reinterpret_cast<float *>(inputs[0]->addr);
  auto *mask = reinterpret_cast<float *>(inputs[1]->addr);
  auto *dx = reinterpret_cast<float *>(outputs[0]->addr);

  DropoutBackward(dy, mask, dx, num_count_, keep_prob_, reinterpret_cast<cudaStream_t>(stream_ptr));

  return true;
}
}  // namespace kernel
}  // namespace mindspore
