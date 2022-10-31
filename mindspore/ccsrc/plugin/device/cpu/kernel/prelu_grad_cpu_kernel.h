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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NN_PRELU_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NN_PRELU_GRAD_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <functional>
#include <string>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class PReLUGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  PReLUGradCpuKernelMod() = default;
  ~PReLUGradCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  using PReLUGradLaunchFunc =
    std::function<bool(PReLUGradCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

 private:
  std::string kernel_name_{};
  PReLUGradLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, PReLUGradLaunchFunc>> func_list_;
  size_t input_length_{0};
  size_t weight_length_{0};
  size_t per_channel_length_{0};
  size_t workspace_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NN_PRELU_GRAD_CPU_KERNEL_H_
