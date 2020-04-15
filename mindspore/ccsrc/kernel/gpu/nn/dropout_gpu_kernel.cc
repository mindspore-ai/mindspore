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

#include "kernel/gpu/nn/dropout_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/dropout_impl.cuh"

namespace mindspore {
namespace kernel {
DropoutGpuFwdKernel::DropoutGpuFwdKernel()
    : cudnn_handle_(nullptr),
      is_null_input_(false),
      num_count_(0),
      drop_prob_(0.0),
      states_init_(false),
      mask_generator_(nullptr) {}

DropoutGpuFwdKernel::~DropoutGpuFwdKernel() { DestroyResource(); }

const std::vector<size_t> &DropoutGpuFwdKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &DropoutGpuFwdKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &DropoutGpuFwdKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool DropoutGpuFwdKernel::Init(const CNodePtr &kernel_node) {
  InitResource();

  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but DropoutGpuFwdKernel needs 1.";
  }

  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  is_null_input_ = CHECK_NULL_INPUT(input_shape);
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }

  num_count_ = 1;
  for (size_t x : input_shape) {
    num_count_ *= x;
  }
  drop_prob_ = GetValue<float>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("drop_prob"));

  InitSizeLists();
  return true;
}

void DropoutGpuFwdKernel::InitResource() {
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
}

void DropoutGpuFwdKernel::DestroyResource() noexcept {}

void DropoutGpuFwdKernel::InitSizeLists() {
  size_t input_size = num_count_ * sizeof(float);
  size_t workspace_size = 0;
  input_size_list_.push_back(input_size);
  output_size_list_.push_back(input_size);  // output size: the same with input size
  output_size_list_.push_back(input_size);  // mask size: the same with input size
  workspace_size_list_.push_back(workspace_size);
}

bool DropoutGpuFwdKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) {
  if (is_null_input_) {
    return true;
  }

  auto *input = reinterpret_cast<float *>(inputs[0]->addr);
  auto *output = reinterpret_cast<float *>(outputs[0]->addr);
  auto *mask = reinterpret_cast<float *>(outputs[1]->addr);

  if (!states_init_) {
    curandCreateGenerator(&mask_generator_, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(mask_generator_, time(NULL));
    states_init_ = true;
  }

  curandGenerateUniform(mask_generator_, mask, num_count_);
  DropoutForward(input, mask, output, num_count_, drop_prob_, reinterpret_cast<cudaStream_t>(stream_ptr));

  return true;
}
}  // namespace kernel
}  // namespace mindspore
