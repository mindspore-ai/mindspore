/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GRAD_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/resize_nearest_neighbor_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ResizeNearestNeighborGradGpuKernel : public GpuKernel {
 public:
  ResizeNearestNeighborGradGpuKernel()
      : align_corners_(false),
        is_null_input_(false),
        shape_size_(0),
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~ResizeNearestNeighborGradGpuKernel() override = default;

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
    int input_size = SizeToInt(input_size_ / sizeof(T));
    float h_scale = Scaling(output_shape_[2], input_shape_[2], align_corners_);
    float w_scale = Scaling(output_shape_[3], input_shape_[3], align_corners_);
    CalResizeNearestNeighborGrad(input_size, input, input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3],
                                 output, output_shape_[0], output_shape_[1], output_shape_[2], output_shape_[3],
                                 align_corners_, h_scale, w_scale, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ResizeNearestNeighbor needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ResizeNearestNeighbor has 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    shape_size_ = input_shape.size();
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ResizeNearestNeighborGradGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }
    if (shape_size_ != RESIZENEARESTNEIGHBORGRAD_DIMENSION) {
      MS_LOG(ERROR) << "Input is " << shape_size_ << "-D, but ResizeNearestNeighbor supports only "
                    << RESIZENEARESTNEIGHBORGRAD_DIMENSION << "-D inputs.";
      return false;
    }
    if (shape_size_ != output_shape.size()) {
      MS_LOG(ERROR) << "The dim of input and output must be same.";
      return false;
    }
    input_size_ = 1;
    for (size_t i = 0; i < shape_size_; i++) {
      input_size_ *= input_shape[i];
      if (input_shape[i] == 0) {
        MS_LOG(ERROR) << "The shape of input has 0.";
        return false;
      }
      input_shape_.push_back(input_shape[i]);
    }
    input_size_ *= sizeof(T);
    output_size_ = 1;
    for (size_t i = 0; i < shape_size_; i++) {
      output_size_ *= output_shape[i];
      if (input_shape[i] == 0) {
        MS_LOG(ERROR) << "The shape of output has 0.";
        return false;
      }
      output_shape_.push_back(output_shape[i]);
    }
    output_size_ *= sizeof(T);
    align_corners_ = GetAttr<bool>(kernel_node, "align_corners");
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  float Scaling(const int in_size, const int out_size, bool align_corners) {
    return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                           : in_size / static_cast<float>(out_size);
  }

  bool align_corners_;
  bool is_null_input_;
  size_t shape_size_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GRAD_GPU_KERNEL_H_
