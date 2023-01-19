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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GATHERND_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GATHERND_CPU_KERNEL_H_

#include <map>
#include <complex>
#include <vector>
#include <utility>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/gather_nd.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnknown = "Unknown";

class GatherNdCpuKernelMod : public NativeCpuKernelMod {
 public:
  GatherNdCpuKernelMod() = default;
  ~GatherNdCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename S, typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using GatherNdFunc = std::function<bool(GatherNdCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, GatherNdFunc>> func_list_;
  GatherNdFunc kernel_func_;

  ShapeVector input_shapes_;
  ShapeVector indices_shapes_;

  std::vector<size_t> dims_;
  std::vector<int> batch_indices_;

  TypeId dtype_{kTypeUnknown};
  std::string kernel_type_{kUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GATHERND_CPU_KERNEL_H_
