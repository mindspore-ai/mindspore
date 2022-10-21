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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ADAMAX_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <utility>
#include <map>
#include <iostream>
#include "mindspore/core/ops/apply_ada_max.h"
#include "kernel/common_utils.h"
#include "include/curand.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adamax_impl.cuh"

namespace mindspore {
namespace kernel {
class AdamaxGpuKernelMod : public NativeGpuKernelMod {
 public:
  AdamaxGpuKernelMod() { ResetResource(); }
  ~AdamaxGpuKernelMod() override = default;

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

  void ResetResource() noexcept {
    is_null_input_ = false;
    t_size_ = DEFAULT_SIZE_;
    s_size_ = DEFAULT_SIZE_;
    g_size_ = DEFAULT_SIZE_;
    input_size_list_.clear();
    output_size_list_.clear();
  }

 private:
  template <typename T, typename S, typename G>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using ApplyAdamaxFunc =
    std::function<bool(AdamaxGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  void InOutputResize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                      const std::vector<KernelTensorPtr> &outputs);

 private:
  constexpr static int64_t DEFAULT_SIZE_ = 4;

  size_t variable_size_{0};
  size_t m_size_{0};
  size_t v_size_{0};
  size_t beta1_power_size_{0};
  size_t learning_rate_size_{0};
  size_t beta1_size_{0};
  size_t beta2_size_{0};
  size_t epsilon_size_{0};
  size_t gradient_size_{0};
  bool is_null_input_{false};

  size_t t_size_{4};
  size_t s_size_{4};
  size_t g_size_{4};
  size_t input_elements_;
  BaseOperatorPtr kernel_ptr_{nullptr};

  std::vector<KernelTensorPtr> outputs_ = {};

  ApplyAdamaxFunc kernel_func_{};
  void *stream_ptr_{nullptr};
  static std::vector<std::pair<KernelAttr, ApplyAdamaxFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_ADAMAX_GPU_KERNEL_H
