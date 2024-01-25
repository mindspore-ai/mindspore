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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ELLIPSIS_TO_SLICE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ELLIPSIS_TO_SLICE_CPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class EllipsisToSliceCpuKernelMod : public NativeCpuKernelMod {
 public:
  EllipsisToSliceCpuKernelMod() = default;
  ~EllipsisToSliceCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  using EllipsisToSliceFunc =
    std::function<bool(EllipsisToSliceCpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;

  static std::vector<std::pair<KernelAttr, EllipsisToSliceFunc>> func_list_;
  EllipsisToSliceFunc kernel_func_;

 private:
  std::vector<std::vector<int64_t>> data_shapes_;
  int64_t end_mask_{0};
  size_t ellipse_position_{0};
  size_t not_ellipse_occupy_dims_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ELLIPSIS_TO_SLICE_CPU_KERNEL_H_
