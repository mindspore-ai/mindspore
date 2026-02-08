/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_TRIU_INDICES_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_TRIU_INDICES_GPU_KERNEL_H_

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mindspore/ops/infer/triu_indices.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/triu_indices_impl.cuh"

namespace mindspore {
namespace kernel {
class TriuIndicesGpuKernelMod : public NativeGpuKernelMod {
 public:
  TriuIndicesGpuKernelMod() { ResetResource(); }
  ~TriuIndicesGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void ResetResource() noexcept;
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using TriuIndicesFunc =
    std::function<bool(TriuIndicesGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  TriuIndicesFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, TriuIndicesFunc>> func_list_;

 private:
  int64_t row_{0};
  int64_t col_{0};
  int64_t offset_{0};
  size_t triu_size_{0};
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_TRIU_INDICES_GPU_KERNEL_H_
