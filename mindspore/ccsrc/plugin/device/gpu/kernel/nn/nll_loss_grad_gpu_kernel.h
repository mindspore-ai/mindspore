/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/loss_with_reduction_impl.cuh"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class NLLLossGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  NLLLossGradGpuKernelMod() { ResetResource(); }
  ~NLLLossGradGpuKernelMod() override = default;

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
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    std::vector<size_t> input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    kernel_node_ = kernel_node;
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "logits");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (input_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of logits cannot be less than 2, but "
                        << "got the " << input_shape.size();
    }
    n_ = static_cast<int>(input_shape[0]);
    c_ = static_cast<int>(input_shape[1]);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    string reduction = GetAttr<string>(kernel_node, "reduction");
    reduction_ = kReductionModeMap[reduction];
    if (reduction_ == ReductionMode::kNone) {
      num_dloss_ = n_;
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    n_ = 0;
    c_ = 0;
    is_null_input_ = false;
    reduction_ = ReductionMode::kMean;  // default value
    num_dloss_ = 1;                     // default size (scalar)
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
  ReductionMode reduction_;
  int n_;
  int c_;
  bool is_null_input_;
  int num_dloss_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_NLL_LOSS_GRAD_GPU_KERNEL_H
