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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PRELU_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PRELU_GRAD_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <functional>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/prelu_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class PReLUGradGpuKernel : public GpuKernel {
 public:
  PReLUGradGpuKernel() = default;
  ~PReLUGradGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto *dy = GetDeviceAddress<T>(inputs, 0);
    auto *x = GetDeviceAddress<T>(inputs, 1);
    auto *w = GetDeviceAddress<T>(inputs, 2);
    auto *dx = GetDeviceAddress<T>(outputs, 0);
    auto *dw = GetDeviceAddress<T>(outputs, 1);
    auto *dw_array = GetDeviceAddress<float>(workspace, 0);

    CalPReLUGrad(input_length_, weight_length_, per_channel_length_, dy, x, w, dx, dw, dw_array,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    ResetResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "ReLUGrad needs 3 inputs, but got " << input_num;
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 2) {
      MS_LOG(ERROR) << "ReLUGrad should have 2 outputs, but got " << input_num;
      return false;
    }

    auto x_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto weight_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
    is_null_input_ = CHECK_NULL_INPUT(x_shape) || CHECK_NULL_INPUT(weight_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'PReLUGradGpuKernel', input is null.";
      InitSizeLists();
      return true;
    }
    input_length_ = std::accumulate(x_shape.begin(), x_shape.end(), size_t(1), std::multiplies<>());
    size_t x_rank = x_shape.size();
    size_t channel_num;
    if (x_rank == 0) {
      channel_num = 1;
      per_channel_length_ = 1;
    } else if (x_rank == 1) {
      channel_num = 1;
      per_channel_length_ = x_shape[0];
    } else {
      channel_num = x_shape[1];
      per_channel_length_ = std::accumulate(x_shape.begin() + 2, x_shape.end(), size_t(1), std::multiplies<>());
    }

    if (weight_shape.size() != 1 || (weight_shape[0] != 1 && weight_shape[0] != channel_num)) {
      MS_LOG(EXCEPTION) << "PReLUGrad requires the rank of weight should be 1, and the elements number should be "
                           "1 or channels number "
                        << channel_num << ", but got weight shape " << weight_shape;
    }
    weight_length_ = weight_shape[0];
    workspace_size_ = weight_length_ * IntToSize(GET_BLOCKS(input_length_) * GET_THREADS) * sizeof(float);
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_length_ = 0;
    weight_length_ = 0;
    per_channel_length_ = 0;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    size_t data_size = sizeof(T);
    input_size_list_.push_back(input_length_ * data_size);
    input_size_list_.push_back(input_length_ * data_size);
    input_size_list_.push_back(weight_length_ * data_size);
    output_size_list_.push_back(input_length_ * data_size);
    output_size_list_.push_back(weight_length_ * data_size);
    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  bool is_null_input_{false};
  size_t input_length_{0};
  size_t weight_length_{0};
  size_t per_channel_length_{0};
  size_t workspace_size_{0};
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PRELU_GRAD_GPU_KERNEL_H_
