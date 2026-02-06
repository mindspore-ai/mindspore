/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_WEIGHTDECAY_SCALE_MOMENTUM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_WEIGHTDECAY_SCALE_MOMENTUM_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/momentum_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class FusedWeightDecayScaleMomentumGpuKernelMod : public NativeGpuKernelMod {
 public:
  FusedWeightDecayScaleMomentumGpuKernelMod() : element_num_(1), is_null_input_(false) {}
  ~FusedWeightDecayScaleMomentumGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *weight_decay = GetDeviceAddress<T>(inputs, 0);
    T *scale = GetDeviceAddress<T>(inputs, 1);
    T *variable = GetDeviceAddress<T>(inputs, 2);
    T *accumulation = GetDeviceAddress<T>(inputs, 3);
    T *learning_rate = GetDeviceAddress<T>(inputs, 4);
    S *gradient = GetDeviceAddress<S>(inputs, 5);
    T *momentum = GetDeviceAddress<T>(inputs, 6);

    auto status =
      FusedWeightDecayScaleMomentum(element_num_, weight_decay, scale, variable, accumulation, learning_rate, gradient,
                                    momentum, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    output_size_list_.clear();
    workspace_size_list_.clear();
    auto variable_shape = inputs[kIndex2]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(variable_shape, kernel_name_, "variable");
    if (is_null_input_) {
      output_size_list_.push_back(element_num_ * sizeof(T));
      return KRET_UNKNOWN_SHAPE;
    }
    element_num_ *= SizeOf(variable_shape);
    output_size_list_.push_back(element_num_ * sizeof(T));
    return KRET_OK;
  }

 private:
  size_t element_num_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_WEIGHTDECAY_SCALE_MOMENTUM_GPU_KERNEL_H_
