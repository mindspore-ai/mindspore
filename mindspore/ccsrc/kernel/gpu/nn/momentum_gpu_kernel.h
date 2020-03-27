/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_MOMENTUM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_MOMENTUM_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/momentum_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class MomentumGpuKernel : public GpuKernel {
 public:
  MomentumGpuKernel()
      : variable_size_(0), accumulation_size_(0), learning_rate_size_(0), gradient_size_(0), momentum_size_(0) {}
  ~MomentumGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              uintptr_t stream_ptr) override {
    T *variable = GetDeviceAddress<T>(inputs, 0);
    T *accumulation = GetDeviceAddress<T>(inputs, 1);
    T *learning_rate = GetDeviceAddress<T>(inputs, 2);
    T *gradient = GetDeviceAddress<T>(inputs, 3);
    T *momentum = GetDeviceAddress<T>(inputs, 4);
    MomentumUpdateVariable(inputs[0]->size / sizeof(T), variable, accumulation, learning_rate, gradient, momentum,
                           reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 5) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but momentum needs 5 inputs.";
      return false;
    }

    variable_size_ = sizeof(T);
    accumulation_size_ = sizeof(T);
    learning_rate_size_ = sizeof(T);
    gradient_size_ = sizeof(T);
    momentum_size_ = sizeof(T);

    auto variable_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < variable_shape.size(); i++) {
      variable_size_ *= variable_shape[i];
    }
    auto accumulation_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    for (size_t i = 0; i < accumulation_shape.size(); i++) {
      accumulation_size_ *= accumulation_shape[i];
    }
    auto gradient_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    for (size_t i = 0; i < gradient_shape.size(); i++) {
      gradient_size_ *= gradient_shape[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(variable_size_);
    input_size_list_.push_back(accumulation_size_);
    input_size_list_.push_back(learning_rate_size_);
    input_size_list_.push_back(gradient_size_);
    input_size_list_.push_back(momentum_size_);
    output_size_list_.push_back(0);
  }

 private:
  size_t variable_size_;
  size_t accumulation_size_;
  size_t learning_rate_size_;
  size_t gradient_size_;
  size_t momentum_size_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_MOMENTUM_GPU_KERNEL_H_
