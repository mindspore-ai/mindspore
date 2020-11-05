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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_REPEAT_ELEMENTS_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_REPEAT_ELEMENTS_GRAD_GPU_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/repeat_elements_grad_impl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class RepeatElementsGradGpuKernel : public GpuKernel {
 public:
  RepeatElementsGradGpuKernel()
      : rep_(1), axis_(0), input_size_(1), output_size_(0), outer_size_(1), repeat_dim_size_(1), inner_size_(1) {}
  ~RepeatElementsGradGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *dx = GetDeviceAddress<T>(outputs, 0);

    CalRepeatElementsGrad(dy, rep_, dx, outer_size_, repeat_dim_size_, inner_size_,
                          reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 1) {
      MS_LOG(EXCEPTION) << input_count << " arguments were provided, but RepeatElementGradGpuKernel expects 1.";
    }

    std::vector<size_t> dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    int dy_dim = dy_shape.size();

    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis_ < 0) {
      axis_ += dy_dim;
    }
    rep_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "rep"));
    if (axis_ >= dy_dim) {
      axis_ = dy_dim - 1;
      rep_ = 1;
    }

    for (int i = 0; i < dy_dim; i++) {
      auto e = dy_shape[i];
      input_size_ *= e;
      input_shape_.push_back(e);
      if (i < axis_) {
        outer_size_ *= e;
      } else if (i > axis_) {
        inner_size_ *= e;
      } else {
        repeat_dim_size_ = e / rep_;
      }
    }

    output_size_ = input_size_ / rep_;
    output_shape_ = input_shape_;
    output_shape_[axis_] /= rep_;

    InitSizeLists();

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  int rep_;
  int axis_;
  size_t input_size_;
  size_t output_size_;
  int outer_size_;
  int repeat_dim_size_;
  int inner_size_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_REPEAT_ELEMENTS_GRAD_GPU_KERNEL_H_
