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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONESLIKE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONESLIKE_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/oneslike_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class OnesLikeGpuKernel : public GpuKernel {
 public:
  OnesLikeGpuKernel() : input_size_(0), output_size_(0) {}
  ~OnesLikeGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    int size = SizeToInt(input_size_ / sizeof(T));
    CalOnesLike(size, input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but oneslike needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but oneslike needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    size_t shape_size = input_shape.size();

    input_size_ = sizeof(T);
    for (size_t i = 0; i < shape_size; i++) {
      input_size_ *= input_shape[i];
    }
    output_size_ = input_size_;
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t input_size_;
  size_t output_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONESLIKE_H_
