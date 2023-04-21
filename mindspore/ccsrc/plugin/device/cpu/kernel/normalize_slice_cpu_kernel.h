/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NORMALIZE_SLICE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NORMALIZE_SLICE_CPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "mindspore/core/ops/tile_size.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class NormalizeSliceInfoCpuKernelMod : public NativeCpuKernelMod {
 public:
  NormalizeSliceInfoCpuKernelMod() = default;
  ~NormalizeSliceInfoCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
              const std::vector<AddressPtr> &workspace) override;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensorPtr> &inputs,
                    const std::vector<kernel::KernelTensorPtr> &outputs,
                    const std::vector<kernel::AddressPtr> &workspace) const;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

  using NormalizeSliceFunc =
    std::function<bool(NormalizeSliceInfoCpuKernelMod *, const std::vector<kernel::KernelTensorPtr> &,
                       const std::vector<kernel::KernelTensorPtr> &, const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, NormalizeSliceFunc>> func_list_;
  NormalizeSliceFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NORMALIZE_SLICE_CPU_KERNEL_H_
