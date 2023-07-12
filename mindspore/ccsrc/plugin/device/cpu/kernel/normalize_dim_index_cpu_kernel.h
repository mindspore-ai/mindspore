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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NORMALIZE_DIM_INDEX_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NORMALIZE_DIM_INDEX_CPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class NormalizeDimIndexCpuKernelMod : public NativeCpuKernelMod {
 public:
  NormalizeDimIndexCpuKernelMod() = default;
  ~NormalizeDimIndexCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &op, const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  bool LaunchKernel(const std::vector<kernel::AddressPtr> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  using NormalizeDimIndexFunc =
    std::function<bool(NormalizeDimIndexCpuKernelMod *, const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, NormalizeDimIndexFunc>> func_list_;
  NormalizeDimIndexFunc kernel_func_;

 private:
  std::vector<std::vector<int64_t>> data_shapes_;
  std::vector<int64_t> tuple_index_types_;
  size_t dim_index_ = 0;
  size_t expand_dims_cnt_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NORMALIZE_DIM_INDEX_CPU_KERNEL_H_
