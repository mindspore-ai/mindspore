/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_GPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include "kernel/gpu/gpu_kernel_factory.h"
#include "mindspore/ops/infer/uniform.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
class UniformGpuKernelMod : public NativeGpuKernelMod {
 public:
  UniformGpuKernelMod() {}
  ~UniformGpuKernelMod() override = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;

  void CheckBernoulliShape();

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using UniformFunc =
    std::function<bool(UniformGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  void *cuda_stream_{nullptr};
  float from_{0.0};
  float to_{1.0};
  uint64_t seed_{0};
  uint64_t seed_offset_{0};
  size_t unit_input_size_{1};
  size_t input_size_{1};
  UniformFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, UniformFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_GPU_KERNEL_H_
