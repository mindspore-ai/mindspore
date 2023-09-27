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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CPU_BLACKMAN_WINDOW_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CPU_BLACKMAN_WINDOW_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class BlackmanWindowCpuKernelMod : public NativeCpuKernelMod {
 public:
  BlackmanWindowCpuKernelMod() = default;
  ~BlackmanWindowCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool BlackmanWindowKernelFunc(const std::vector<kernel::KernelTensor *> &inputs,
                                const std::vector<kernel::KernelTensor *> &workspace,
                                const std::vector<kernel::KernelTensor *> &outputs) const;
  bool periodic_{true};
  TypeId input_dtype{kTypeUnknown};
  using BlackmanWindowFunc =
    std::function<bool(BlackmanWindowCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, BlackmanWindowFunc>> func_list_;
  BlackmanWindowFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CPU_BLACKMAN_WINDOW_CPU_KERNEL_H_
