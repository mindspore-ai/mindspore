/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_REVERSE_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_REVERSE_V2_GPU_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/reverse_v2_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ReverseV2GpuKernel : public GpuKernel {
 public:
  ReverseV2GpuKernel() { ResetResource(); }
  ~ReverseV2GpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_device = GetDeviceAddress<T>(inputs, 0);
    T *output_device = GetDeviceAddress<T>(outputs, 0);
    size_t *input_shape_device = GetDeviceAddress<size_t>(workspace, 0);
    int64_t *strides_device = GetDeviceAddress<int64_t>(workspace, 1);
    int64_t *axis_device = GetDeviceAddress<int64_t>(workspace, 2);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape_device, &input_shape_[0], workspace_size_list_[0],
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for input_shape_ failed");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(strides_device, &strides_[0], workspace_size_list_[1],
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for strides_ failed");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(axis_device, &axis_[0], workspace_size_list_[2], cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for axis_ failed");

    CalReverseV2(input_device, output_device, input_shape_device, strides_device, axis_device, input_size_,
                 axis_.size(), reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 1) {
      MS_LOG(ERROR) << input_count << " inputs were provided, but ReverseV2GpuKernel expects 1.";
      return false;
    }

    size_t output_count = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_count != 1) {
      MS_LOG(ERROR) << "Number of outputs is " << output_count << ", but should be 1 for ReverseV2GpuKernel.";
      return false;
    }

    input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ReverseV2GpuKernel', input is null.";
      InitSizeLists();
      return true;
    }
    input_rank_ = input_shape_.size();
    if (input_rank_ < 1) {
      MS_LOG(EXCEPTION) << "For 'ReverseV2GpuKernel', the rank of input cannot be less than 1, bot got " << input_rank_;
    }
    input_size_ = 1;
    for (size_t i = 0; i < input_rank_; i++) {
      input_size_ *= input_shape_[i];
    }

    strides_.resize(input_rank_);
    strides_[input_rank_ - 1] = 1;
    for (int32_t i = input_rank_ - 2; i >= 0; i--) {
      strides_[i] = static_cast<int64_t>(input_shape_[i + 1]) * strides_[i + 1];
    }

    axis_ = GetAttr<std::vector<int64_t>>(kernel_node, "axis");
    if (axis_.size() < 1) {
      MS_LOG(EXCEPTION) << "For 'ReverseV2GpuKernel', the rank of axis cannot be less than 1, bot got " << axis_.size();
    }
    for (int64_t &dimension : axis_) {
      if (dimension < 0) {
        dimension += input_rank_;
      }
    }

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    input_rank_ = 0;
    is_null_input_ = false;
    input_shape_.clear();
    strides_.clear();
    axis_.clear();

    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    size_t input_bytes = input_size_ * sizeof(T);
    input_size_list_.push_back(input_bytes);
    output_size_list_.push_back(input_bytes);
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
    workspace_size_list_.push_back(input_rank_ * sizeof(int64_t));
    workspace_size_list_.push_back(axis_.size() * sizeof(int64_t));
  }

 private:
  size_t input_size_;
  size_t input_rank_;
  std::vector<size_t> input_shape_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> axis_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_REVERSE_V2_GPU_KERNEL_H_
