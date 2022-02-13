/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/resize_nearest_neighbor_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ResizeNearestNeighborGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeNearestNeighborGpuKernelMod()
      : align_corners_(false),
        is_null_input_(false),
        shape_size_(0),
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~ResizeNearestNeighborGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    int size = SizeToInt(output_size_ / sizeof(T));
    float h_scale = Scaling(input_shape_[2], output_shape_[2], align_corners_);
    float w_scale = Scaling(input_shape_[3], output_shape_[3], align_corners_);
    CalResizeNearestNeighbor(size, input, input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3], output,
                             output_shape_[0], output_shape_[1], output_shape_[2], output_shape_[3], align_corners_,
                             h_scale, w_scale, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    shape_size_ = input_shape.size();
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (shape_size_ != RESIZENEARESTNEIGHBOR_DIMENSION) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input should be "
                        << RESIZENEARESTNEIGHBOR_DIMENSION << ", but got " << shape_size_;
    }
    if (shape_size_ != output_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name
                        << "', the dimension of input and output should be the same, but got the dimension of input: "
                        << shape_size_ << ", the dimension of output: " << output_shape.size();
    }
    input_size_ = 1;
    for (size_t i = 0; i < shape_size_; i++) {
      input_size_ *= input_shape[i];
      input_shape_.push_back(input_shape[i]);
    }
    input_size_ *= sizeof(T);
    output_size_ = 1;
    for (size_t i = 0; i < shape_size_; i++) {
      output_size_ *= output_shape[i];
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
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GPU_KERNEL_H_
