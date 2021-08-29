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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEAVGPOOL2D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEAVGPOOL2D_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/adaptive_avg_pool2d_grad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kAdaptiveAvgPool2dGradInputNum = 2;
constexpr size_t kAdaptiveAvgPool2dGradMinRank = 2;

template <typename T>
class AdaptiveAvgPool2DGradKernel : public GpuKernel {
 public:
  AdaptiveAvgPool2DGradKernel()
      : input_size_(0),
        output_size_(0),
        input_height_(0),
        input_width_(0),
        output_height_(0),
        output_width_(0),
        size_(0) {}
  ~AdaptiveAvgPool2DGradKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *dy_addr = GetDeviceAddress<T>(inputs, 1);
    T *dx_addr = GetDeviceAddress<T>(outputs, 0);

    ApplyAdaptiveAvgPool2DGrad(size_, input_height_, input_width_, output_height_, output_width_, dy_addr, dx_addr,
                               reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != kAdaptiveAvgPool2dGradInputNum) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but AdaptiveAvgPool2DGrad needs "
                    << kAdaptiveAvgPool2dGradInputNum << " inputs.";
      return false;
    }

    input_size_ = sizeof(T);
    output_size_ = sizeof(T);

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);  // dy
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);         // dx

    auto input_rank = input_shape.size();
    auto output_rank = output_shape.size();
    if (input_rank < kAdaptiveAvgPool2dGradMinRank || output_rank < kAdaptiveAvgPool2dGradMinRank) {
      MS_LOG(ERROR) << "The input or output should have rank at least 2.";
      return false;
    }
    input_height_ = static_cast<uint>(input_shape[input_rank - 2]);
    input_width_ = static_cast<uint>(input_shape[input_rank - 1]);
    size_ = static_cast<uint>(input_rank == (kAdaptiveAvgPool2dGradMinRank + 1) ? input_shape[0]
                                                                                : input_shape[0] * input_shape[1]);
    for (uint i = 0; i < input_rank; i++) {
      input_size_ *= input_shape[i];
    }

    output_height_ = static_cast<uint>(output_shape[output_rank - 2]);
    output_width_ = static_cast<uint>(output_shape[output_rank - 1]);
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t input_size_;
  size_t output_size_;
  uint input_height_;
  uint input_width_;
  uint output_height_;
  uint output_width_;
  uint size_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAPTIVEAVGPOOL2D_GPU_KERNEL_H_
