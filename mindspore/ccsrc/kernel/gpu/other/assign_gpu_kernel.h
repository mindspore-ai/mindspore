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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_ASSIGN_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_ASSIGN_GPU_KERNEL_H

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class AssignGpuKernel : public GpuKernel {
 public:
  AssignGpuKernel() : input_size_(0) {}
  ~AssignGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    T *var = GetDeviceAddress<T>(inputs, 0);
    T *value = GetDeviceAddress<T>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(var, value, input_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemxcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(output, value, input_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemxcpyAsync failed.");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_size_ = sizeof(T);
    for (size_t x : shape) {
      input_size_ = input_size_ * x;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but AssignGpuKernel needs 2 output.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but AssignGpuKernel needs 1 output.";
      return false;
    }
    return true;
  }

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  size_t input_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_ASSIGN_GPU_KERNEL_H
