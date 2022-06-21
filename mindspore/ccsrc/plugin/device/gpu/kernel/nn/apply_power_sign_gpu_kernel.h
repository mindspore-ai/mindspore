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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_APPLY_ADD_SIGN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_APPLY_ADD_SIGN_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <utility>
#include <map>
#include <iostream>
#include "mindspore/core/ops/apply_add_sign.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_power_sign_impl.cuh"

namespace mindspore {
namespace kernel {
class ApplyPowerSignGpuKernelMod : public NativeGpuKernelMod {
 public:
  ApplyPowerSignGpuKernelMod() { ResetResource(); }
  ~ApplyPowerSignGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

  void ResetResource() noexcept;

 private:
  template <typename T, typename S, typename G>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using ApplyPowerSignFunc = std::function<bool(ApplyPowerSignGpuKernelMod *, const std::vector<AddressPtr> &,
                                                const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;

 private:
  size_t t_size_{1};
  size_t s_size_{1};
  size_t g_size_{1};
  size_t t_elements_;
  size_t s_elements_;
  size_t g_elements_;
  bool is_null_input_{false};
  void *stream_ptr_{nullptr};
  ApplyPowerSignFunc kernel_func_{};
  std::optional<bool> is_input_dynamic_shape_{};
  static std::vector<std::pair<KernelAttr, ApplyPowerSignFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif
