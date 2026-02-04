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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FAST_GELU_GRAD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FAST_GELU_GRAD_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cuda_impl/cuda_ops/fast_gelu_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kUnknown = "Unknown";

class FastGeLUGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  FastGeLUGradGpuKernelMod() = default;
  explicit FastGeLUGradGpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~FastGeLUGradGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

 private:
  using FastGeluGradLaunchFunc =
    std::function<bool(FastGeLUGradGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, FastGeluGradLaunchFunc>> func_list_;
  FastGeluGradLaunchFunc kernel_func_;
  std::string kernel_type_{kUnknown};
  void *cuda_stream_{nullptr};
  std::vector<size_t> input_shape_;
  size_t unit_size_{1};
  bool is_null_input_{false};
  size_t input_elements_{};
  const size_t max_dims_{7};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FAST_GELU_GRAD_KERNEL_H_
