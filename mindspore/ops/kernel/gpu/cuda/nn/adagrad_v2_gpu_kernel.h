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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAGRAD_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAGRAD_V2_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <functional>
#include <map>
#include <utility>
#include <memory>
#include <algorithm>
#include <iostream>
#include "mindspore/ops/infer/apply_adagrad_v2.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/curand.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/adagrad_v2_impl.cuh"

namespace mindspore {
namespace kernel {
class AdagradV2GpuKernelMod : public NativeGpuKernelMod {
 public:
  AdagradV2GpuKernelMod() = default;
  ~AdagradV2GpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

  void ResetResource() noexcept {
    is_null_input_ = false;
    t_size_ = DEFAULT_SIZE_;
    s_size_ = DEFAULT_SIZE_;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using ApplyAdagradV2Func =
    std::function<bool(AdagradV2GpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  void InOutputResize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

 private:
  constexpr static int64_t DEFAULT_SIZE_ = 4;

  float epsilon_;
  bool update_slots_;

  int64_t variable_size_{0};
  int64_t accumulation_size_{0};
  int64_t learning_rate_size_{0};
  int64_t gradient_size_{0};
  bool is_null_input_{false};

  int64_t t_size_{4};
  int64_t s_size_{4};
  int64_t input_elements_;
  std::vector<KernelTensor *> outputs_ = {};

  ApplyAdagradV2Func kernel_func_{};
  void *stream_ptr_{nullptr};
  static std::vector<std::pair<KernelAttr, ApplyAdagradV2Func>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_ADAGRAD_V2_GPU_KERNEL_H
