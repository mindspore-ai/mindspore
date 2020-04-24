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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_TRANSPOSE_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_TRANSPOSE_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/transpose_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class TransposeGpuFwdKernel : public GpuKernel {
 public:
  TransposeGpuFwdKernel() : shape_size_(0), input_size_(0), output_size_(0), workspace_size_(0) {}
  ~TransposeGpuFwdKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    int *input_shape = GetDeviceAddress<int>(workspace, 0);
    int *input_axis = GetDeviceAddress<int>(workspace, 1);
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(input_shape, &input_shape_[0], workspace_size_, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_shape failed");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(input_axis, &input_axis_[0], workspace_size_, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_axis failed");
    int size = SizeToInt(input_size_ / sizeof(T));
    CalTranspose(size, input, input_shape, input_axis, SizeToInt(shape_size_), output,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but transpose needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but transpose needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    shape_size_ = input_shape.size();
    if (shape_size_ > TRANSPOSE_MAX_DIMENSION) {
      MS_LOG(EXCEPTION) << "Input is " << shape_size_ << "-D, but transpose supports max " << TRANSPOSE_MAX_DIMENSION
                        << "-D inputs.";
    }

    input_size_ = 1;
    for (size_t i = 0; i < shape_size_; i++) {
      input_size_ *= input_shape[i];
      input_shape_.push_back(input_shape[i]);
    }
    input_size_ *= sizeof(T);
    output_size_ = input_size_;
    auto perm = GetAttr<std::vector<int>>(kernel_node, "perm");
    for (size_t j = 0; j < perm.size(); j++) {
      input_axis_.push_back(perm[j]);
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    workspace_size_ = shape_size_ * sizeof(int);
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
    return;
  }

 private:
  std::vector<int> input_shape_;
  std::vector<int> input_axis_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t shape_size_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_TRANSPOSE_H_
