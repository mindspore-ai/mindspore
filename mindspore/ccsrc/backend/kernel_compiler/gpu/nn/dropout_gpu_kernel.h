/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/dropout_impl.cuh"
#include "include/curand.h"

namespace mindspore {
namespace kernel {
template <typename T>
class DropoutGpuFwdKernel : public GpuKernel {
 public:
  DropoutGpuFwdKernel() { ResetResource(); }
  ~DropoutGpuFwdKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    T *mask = GetDeviceAddress<T>(outputs, 1);
    float *mask_f = GetDeviceAddress<float>(workspace, 0);

    if (!states_init_) {
      CHECK_CURAND_RET_WITH_EXCEPT(curandCreateGenerator(&mask_generator_, CURAND_RNG_PSEUDO_DEFAULT),
                                   "Failed to create generator");
      CHECK_CURAND_RET_WITH_EXCEPT(curandSetPseudoRandomGeneratorSeed(mask_generator_, time(NULL)),
                                   "Failed to SetPseudoRandomGeneratorSeed");
      MS_EXCEPTION_IF_NULL(mask_generator_);
      states_init_ = true;
    }
    CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(mask_generator_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "Failed to set stream for generator");
    // curandGen only support float or double for mask.
    CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateUniform(mask_generator_, mask_f, num_count_),
                                 "Failed to generate uniform");
    DropoutForward(input, mask, output, mask_f, num_count_, keep_prob_, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
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
    keep_prob_ = GetAttr<float>(kernel_node, "keep_prob");

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    is_null_input_ = false;
    num_count_ = 0;
    keep_prob_ = 0.0;
    states_init_ = false;
    mask_generator_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitResource() override { cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle(); }

  void InitSizeLists() override {
    size_t input_size = num_count_ * sizeof(T);
    input_size_list_.push_back(input_size);
    output_size_list_.push_back(input_size);                     // output size: the same with input size
    output_size_list_.push_back(input_size);                     // mask size: the same with input size
    workspace_size_list_.push_back(num_count_ * sizeof(float));  // temp mask_f for curandGen
  }

 private:
  cudnnHandle_t cudnn_handle_;
  bool is_null_input_;
  size_t num_count_;
  float keep_prob_;
  bool states_init_;
  curandGenerator_t mask_generator_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT_GPU_KERNEL_H_
