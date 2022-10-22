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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RAGGED_RANGE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RAGGED_RANGE_CPU_KERNEL_H_
#include <functional>
#include <vector>
#include <memory>
#include <cmath>
#include <type_traits>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class RaggedRangeCpuKernelMod : public NativeCpuKernelMod {
 public:
  RaggedRangeCpuKernelMod() = default;
  ~RaggedRangeCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
              const std::vector<kernel::AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  TypeId input_type_{kTypeUnknown};
  bool broadcast_starts_{false};
  bool broadcast_limits_{false};
  bool broadcast_deltas_{false};
  TypeId tsplits_type_{kTypeUnknown};
  std::vector<int> in_sizes_;
  template <typename T, typename TSPLITS>
  void RaggedRangeLaunch(const size_t nrows, const std::vector<kernel::AddressPtr> &inputs, bool broadcast_starts,
                         bool broadcast_limits, bool broadcast_deltas,
                         const std::vector<kernel::AddressPtr> &outputs) const;
  template <typename T, typename TSPLITS>
  TSPLITS RangeSize(T start, T limit, T delta) const;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RAGGED_RANGE_CPU_KERNEL_H_
