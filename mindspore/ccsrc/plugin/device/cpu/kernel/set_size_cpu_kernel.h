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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SET_SIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SET_SIZE_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SetSizeCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  SetSizeCpuKernelMod() = default;
  ~SetSizeCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool IndicesValid(int64_t n, const std::vector<kernel::AddressPtr> &inputs) const;

  template <typename T>
  bool SetSizeCompute(const std::vector<kernel::AddressPtr> &inputs,
                      const std::vector<kernel::AddressPtr> &outputs) const;

  ShapeVector output_shape_;
  ShapeVector shape_;
  bool validate_indices_{true};
  int64_t dims_{0};
  TypeId val_dtype_{kTypeUnknown};
  size_t values_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SET_SIZE_CPU_KERNEL_H_
