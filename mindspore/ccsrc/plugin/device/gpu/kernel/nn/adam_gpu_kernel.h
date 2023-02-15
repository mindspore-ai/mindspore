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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adam_impl.cuh"
namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 10;
class AdamGpuKernelMod : public NativeGpuKernelMod {
 public:
  AdamGpuKernelMod()
      : variable_size_(0),
        m_size_(0),
        v_size_(0),
        beta1_power_size_(0),
        beta2_power_size_(0),
        learning_rate_size_(0),
        beta1_size_(0),
        beta2_size_(0),
        epsilon_size_(0),
        gradient_size_(0),
        is_null_input_(false),
        kernel_name_("Adam") {}

  ~AdamGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using AdamLaunchFunc =
    std::function<bool(AdamGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  AdamLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, AdamLaunchFunc>> func_list_;
  size_t input_elements_;
  int64_t batch_rank_{0};
  int64_t batch_size_{1};
  size_t variable_size_;
  size_t m_size_;
  size_t v_size_;
  size_t beta1_power_size_;
  size_t beta2_power_size_;
  size_t learning_rate_size_;
  size_t beta1_size_;
  size_t beta2_size_;
  size_t epsilon_size_;
  size_t gradient_size_;
  bool is_null_input_;
  bool use_nesterov_ = false;
  bool use_locikng_ = false;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_ADAM_GPU_KERNEL_H_
