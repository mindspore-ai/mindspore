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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_LOGNORMALREVERSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_LOGNORMALREVERSE_GPU_KERNEL_H_

#include <utility>
#include <vector>
#include <functional>
#include <ctime>
#include <map>
#include "include/curand.h"
#include "mindspore/ops/infer/log_normal_reverse.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/log_normal_reverse_impl.cuh"

namespace mindspore {
namespace kernel {
class LogNormalReverseGpuKernelMod : public NativeGpuKernelMod {
 public:
  LogNormalReverseGpuKernelMod() = default;
  ~LogNormalReverseGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  void ResetResource() noexcept;

  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

 private:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)};
    return support_list;
  }

  TypeId input_dtype_{kTypeUnknown};
  TypeId output_dtype_{kTypeUnknown};
  cudaStream_t cuda_stream_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> input_shape_;
  float input_mean_;
  float input_std_;
  void *stream_ptr_{nullptr};
  size_t unit_size_{1};
  size_t input_elements_;
  bool states_init_{false};
  uint64_t seed_{0};
  curandGenerator_t mask_generator_;
  using LogNormalReverseFunc =
    std::function<bool(LogNormalReverseGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, LogNormalReverseFunc>> func_list_;
  LogNormalReverseFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_LOGNORMALREVERSE_GPU_KERNEL_H_
