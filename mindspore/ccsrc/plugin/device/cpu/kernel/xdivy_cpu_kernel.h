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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_XDIVY_CPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_XDIVY_CPU_KERNEL_H
#include <complex>
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class XdivyCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<XdivyCpuKernelMod> {
 public:
  XdivyCpuKernelMod() { ResetResource(); }
  ~XdivyCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 protected:
  void ResetResource() noexcept {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    index_listx_.clear();
    index_listy_.clear();
  }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  std::vector<size_t> index_listx_{};
  std::vector<size_t> index_listy_{};
  bool is_need_broadcast_{false};
  bool has_null_input_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_XDIVY_CPU_KERNEL_H
