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
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class NormalizeSliceInfoCpuKernelMod : public NativeCpuKernelMod {
 public:
  NormalizeSliceInfoCpuKernelMod() = default;
  ~NormalizeSliceInfoCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &op, const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                    const std::vector<kernel::AddressPtr> &outputs) const;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

  using NormalizeSliceFunc =
    std::function<bool(NormalizeSliceInfoCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, NormalizeSliceFunc>> func_list_;
  NormalizeSliceFunc kernel_func_;

 private:
  ShapeVector data_shape_;
  size_t index_axis_ = 0;
  int64_t expand_dims_mask_ = 0;
  std::vector<int64_t> tuple_index_types_;
  std::vector<int64_t> init_by_none_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NORMALIZE_SLICE_CPU_KERNEL_H_
