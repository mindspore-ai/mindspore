/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CORRELATE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CORRELATE_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cast_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/reverse_v2_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/correlate_impl.cuh"

namespace mindspore {
namespace kernel {
class CorrelateGpuKernelMod : public NativeGpuKernelMod {
 public:
  CorrelateGpuKernelMod() { ResetResource(); }
  ~CorrelateGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void ResetResource() noexcept {
    a_size_ = 0;
    v_size_ = 0;
    a_ge_v_ = true;
    long_size_ = 0;
    short_size_ = 0;
    data_unit_size_ = 1;
    copy_start_idx_ = 0;
    mode_type_ = mindspore::PadMode::VALID;
    padded_long_size_ = 0;
    out_size_ = 0;
    kernel_func_ = nullptr;
    cuda_stream_ = nullptr;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T_in, typename T_out>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  template <typename T>
  bool LaunchKernelComplex(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs);
  using CorrelateFunc =
    std::function<bool(CorrelateGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  int64_t a_size_{0};
  int64_t v_size_{0};
  bool a_ge_v_{true};
  int64_t long_size_{0};
  int64_t short_size_{0};
  size_t data_unit_size_{1};
  int64_t copy_start_idx_{0};
  mindspore::PadMode mode_type_{mindspore::PadMode::VALID};
  int64_t padded_long_size_{0};
  int64_t out_size_{0};
  CorrelateFunc kernel_func_{nullptr};
  void *cuda_stream_{nullptr};
  static std::vector<std::pair<KernelAttr, CorrelateFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CORRELATE_GPU_KERNEL_H_
