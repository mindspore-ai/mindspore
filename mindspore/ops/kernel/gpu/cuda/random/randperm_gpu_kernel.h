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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDPERM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDPERM_GPU_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include <map>
#include <utility>

#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class RandpermGpuKernelMod : public NativeGpuKernelMod {
 public:
  // initialize rng here to minimize how many times we seed it.
  RandpermGpuKernelMod() : rng_(std::random_device()()) {}

  ~RandpermGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using RandpermGpuLaunchFunc = std::function<bool(RandpermGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                                   const std::vector<kernel::KernelTensor *> &,
                                                   const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  RandpermGpuLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, RandpermGpuLaunchFunc>> func_list_;
  std::mt19937 rng_;
  size_t max_length_{1};
  int64_t pad_{-1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDPERM_GPU_KERNEL_H_
