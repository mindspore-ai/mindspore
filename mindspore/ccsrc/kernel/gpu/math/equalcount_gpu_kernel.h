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

#ifndef MINDSPORE_EQUALCOUNT_GPU_KERNEL_H
#define MINDSPORE_EQUALCOUNT_GPU_KERNEL_H

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/equalcount_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class EqualCountGpuKernel : public GpuKernel {
 public:
  EqualCountGpuKernel() : input_size_(0), output_size_(0), workspace_size_(0) {}
  ~EqualCountGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input1 = GetDeviceAddress<T>(inputs, 0);
    T *input2 = GetDeviceAddress<T>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);
    int size = SizeToInt(input_size_ / sizeof(T));
    CalEqualCount(size, input1, input2, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but equalcount needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but equalcount needs 1 output.";
      return false;
    }

    output_size_ = sizeof(T);
    input_size_ = sizeof(T);
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif
