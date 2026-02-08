/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MULTINOMIAL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MULTINOMIAL_GPU_KERNEL_H_

#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <functional>
#include <algorithm>
#include <utility>
#include "mindspore/ops/infer/multinomial.h"
#include "plugin/gpu/res_manager/mem_manager/gpu_memory_allocator.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/multinomial_impl.cuh"

namespace mindspore {
namespace kernel {
class MultinomialGpuKernelMod : public NativeGpuKernelMod {
 public:
  MultinomialGpuKernelMod() = default;
  ~MultinomialGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  size_t distributions_{0};
  size_t categories_{0};
  uint64_t seed_{0};
  uint64_t seed_offset_{0};
  bool rand_state_init_{false};
  curandState *rand_state_{nullptr};

  template <typename T, typename S>
  void LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs, void *stream_ptr);
  using LaunchFunc = std::function<void(MultinomialGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &, void *stream_ptr)>;
  LaunchFunc launch_func_;

  static std::vector<std::pair<KernelAttr, LaunchFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MULTINOMIAL_GPU_KERNEL_H_
