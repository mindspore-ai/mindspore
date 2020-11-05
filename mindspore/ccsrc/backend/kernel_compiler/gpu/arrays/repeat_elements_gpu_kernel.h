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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_REPEAT_ELEMENTS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_REPEAT_ELEMENTS_GPU_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/repeat_elements_impl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class RepeatElementsGpuKernel : public GpuKernel {
 public:
  RepeatElementsGpuKernel() : rep_(1), axis_(0), input_size_(1), output_size_(0) {}
  ~RepeatElementsGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_device_address = GetDeviceAddress<T>(inputs, 0);
    T *output_device_address = GetDeviceAddress<T>(outputs, 0);

    switch (input_dim_) {
      case 1:
        CalRepeatElements1d(input_device_address, rep_, axis_, output_device_address, output_size_,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case 2:
        CalRepeatElements2d(input_device_address, input_shape_[1], rep_, axis_, output_device_address, output_shape_[1],
                            output_size_, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case 3:
        CalRepeatElements3d(input_device_address, input_shape_[1], input_shape_[2], rep_, axis_, output_device_address,
                            output_shape_[1], output_shape_[2], output_size_,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case 4:
        CalRepeatElements4d(input_device_address, input_shape_[1], input_shape_[2], input_shape_[3], rep_, axis_,
                            output_device_address, output_shape_[1], output_shape_[2], output_shape_[3], output_size_,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case 5:
        CalRepeatElements5d(input_device_address, input_shape_[1], input_shape_[2], input_shape_[3], input_shape_[4],
                            rep_, axis_, output_device_address, output_shape_[1], output_shape_[2], output_shape_[3],
                            output_shape_[4], output_size_, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      default:
        int *input_shape_device_address = GetDeviceAddress<int>(workspace, 0);
        int *output_shape_device_address = GetDeviceAddress<int>(workspace, 1);
        int *input_shape_cumulative_product_device_address = GetDeviceAddress<int>(workspace, 2);
        CHECK_CUDA_RET_WITH_EXCEPT(
          cudaMemcpyAsync(input_shape_device_address, input_shape_.data(), workspace_size_list_[0],
                          cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
          "cudaMemcpyAsync input_shape failed");
        CHECK_CUDA_RET_WITH_EXCEPT(
          cudaMemcpyAsync(output_shape_device_address, output_shape_.data(), workspace_size_list_[1],
                          cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
          "cudaMemcpyAsync output_shape failed");
        CHECK_CUDA_RET_WITH_EXCEPT(
          cudaMemcpyAsync(input_shape_cumulative_product_device_address, input_shape_cumulative_product_.data(),
                          workspace_size_list_[2], cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
          "cudaMemcpyAsync input_shape_cumulative_product_device_address failed");

        CalRepeatElements(input_device_address, input_dim_, input_shape_device_address,
                          input_shape_cumulative_product_device_address, rep_, axis_, output_device_address,
                          output_shape_device_address, output_size_, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 1) {
      MS_LOG(EXCEPTION) << input_count << " arguments were provided, but RepeatElementsGpuKernel expects 1.";
    }

    std::vector<size_t> temp_input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_dim_ = temp_input_shape.size();
    for (size_t e : temp_input_shape) {
      input_size_ *= e;
      input_shape_.push_back(e);
    }

    int cumulative_product = 1;
    for (size_t i = input_dim_ - 1; i > 0; i--) {
      cumulative_product *= input_shape_[i];
      input_shape_cumulative_product_.push_back(cumulative_product);
    }
    std::reverse(input_shape_cumulative_product_.begin(), input_shape_cumulative_product_.end());

    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis_ < 0) {
      axis_ += input_dim_;
    }

    rep_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "rep"));
    output_size_ = input_size_ * rep_;
    output_shape_ = input_shape_;
    output_shape_[axis_] *= rep_;

    InitSizeLists();

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(T));

    // workspaces for input shape, output shape and cumulative sum
    workspace_size_list_.push_back(input_dim_ * sizeof(int));
    workspace_size_list_.push_back(input_dim_ * sizeof(int));
    workspace_size_list_.push_back((input_dim_ - 1) * sizeof(int));
  }

 private:
  int rep_;
  int axis_;
  int input_dim_;
  std::vector<int> input_shape_;
  std::vector<int> input_shape_cumulative_product_;
  std::vector<int> output_shape_;

  size_t input_size_;
  size_t output_size_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_REPEAT_ELEMENTS_GPU_KERNEL_H_
