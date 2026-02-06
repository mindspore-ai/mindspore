/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the
 * "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the
 * License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in
 * writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions
 * and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_PROXIMAL_ADAGRAD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_PROXIMAL_ADAGRAD_KERNEL_H_

#include <vector>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_apply_proximal_adagrad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 7;
template <typename T>
class SparseApplyProximalAdagradKernelMod : public NativeGpuKernelMod {
 public:
  SparseApplyProximalAdagradKernelMod() = default;
  ~SparseApplyProximalAdagradKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    T *variable = GetDeviceAddress<T>(inputs, 0);
    T *accumulation = GetDeviceAddress<T>(inputs, 1);
    T *learning_rate = GetDeviceAddress<T>(inputs, 2);
    T *l1_regularization = GetDeviceAddress<T>(inputs, 3);
    T *l2_regularization = GetDeviceAddress<T>(inputs, 4);
    T *gradient = GetDeviceAddress<T>(inputs, 5);
    int *indices = GetDeviceAddress<int>(inputs, 6);
    T *variable_out = GetDeviceAddress<T>(outputs, 0);
    T *accumulation_out = GetDeviceAddress<T>(outputs, 1);

    auto status = CalSparseApplyProximalAdagrad(
      inputs[0]->size() / sizeof(T), indices_size_, learning_rate, l1_regularization, l2_regularization, gradient,
      indices, variable, accumulation, variable_out, accumulation_out, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    if (inputs.empty() || outputs.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
      return false;
    }
    if (inputs.size() != INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << INPUT_NUM << ", but got "
                        << inputs.size();
    }
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    auto indices_shape = inputs[kIndex6]->GetShapeVector();
    indices_size_ = SizeOf(indices_shape);
    return KRET_OK;
  }

 private:
  size_t indices_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_PROXIMAL_ADAGRAD_KERNEL_H_
