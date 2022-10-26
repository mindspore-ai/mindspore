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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SET_SIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SET_SIZE_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <utility>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SetSizeCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<SetSizeCpuKernelMod> {
 public:
  SetSizeCpuKernelMod() = default;
  ~SetSizeCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

 private:
  bool IndicesValid(int64_t n, const std::vector<kernel::AddressPtr> &inputs) const;

  ShapeVector output_shape_;
  ShapeVector shape_;
  bool validate_indices_{true};
  int64_t dims_{0};
  TypeId val_dtype_{kTypeUnknown};
  size_t values_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SET_SIZE_CPU_KERNEL_H_
