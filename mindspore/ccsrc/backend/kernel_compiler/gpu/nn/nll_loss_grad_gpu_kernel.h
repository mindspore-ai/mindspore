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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_NLL_LOSS_GRAD_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_NLL_LOSS_GRAD_GPU_KERNEL_H

#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/loss_with_reduction_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class NLLLossGradGpuKernel : public GpuKernel {
 public:
  NLLLossGradGpuKernel() { ResetResource(); }
  ~NLLLossGradGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_device = GetDeviceAddress<T>(inputs, 0);
    T *dloss_device = GetDeviceAddress<T>(inputs, 1);
    int32_t *target_device = GetDeviceAddress<int32_t>(inputs, 2);  // nll_loss_grad only supports int32 target
    S *weight_device = GetDeviceAddress<S>(inputs, 3);
    S *total_weight_device = GetDeviceAddress<S>(inputs, 4);

    T *dinput_device = GetDeviceAddress<T>(outputs, 0);

    NLLLossGrad(n_, c_, reduction_, input_device, target_device, weight_device, total_weight_device, dloss_device,
                dinput_device, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    std::vector<size_t> input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'NllLossGradGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    if (input_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For 'NllLossGradGpuKernel', the rank of input cannot less than 2, but got "
                        << input_shape.size();
    }
    n_ = static_cast<int>(input_shape[0]);
    c_ = static_cast<int>(input_shape[1]);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    string reduction = GetAttr<string>(kernel_node, "reduction");

    // if reduction is not 'none', tmp_nll is (N,) size
    if (reduction == "none") {
      reduction_ = 0;
      num_dloss_ = n_;  // dloss is a vector
    } else if (reduction == "sum") {
      reduction_ = 2;
    } else {
      // reduction = 'mean'
      reduction_ = 1;
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    n_ = 0;
    c_ = 0;
    is_null_input_ = false;
    reduction_ = 1;  // default value
    num_dloss_ = 1;  // default size (scalar)
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));  // input tensor with shape (N, C)
    input_size_list_.push_back(num_dloss_ * sizeof(T));   // dloss tensor (either scalar or size N)
    input_size_list_.push_back(n_ * sizeof(int32_t));     // target tensor with shape (N)
    input_size_list_.push_back(c_ * sizeof(S));           // weight tensor with shape (C)
    input_size_list_.push_back(sizeof(S));                // total_weight scalar

    output_size_list_.push_back(input_size_ * sizeof(T));  // dinput
  }

 private:
  size_t input_size_;
  int reduction_;
  int n_;
  int c_;
  bool is_null_input_;
  int num_dloss_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_NLL_LOSS_GRAD_GPU_KERNEL_H
