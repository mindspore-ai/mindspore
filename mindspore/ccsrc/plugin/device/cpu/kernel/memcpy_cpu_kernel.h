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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESHAPE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESHAPE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MemcpyCpuKernelMod : public NativeCpuKernelMod {
 public:
  MemcpyCpuKernelMod() = default;
  explicit MemcpyCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~MemcpyCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &,
            const std::vector<KernelTensorPtr> &) override {
    MS_EXCEPTION_IF_NULL(base_operator);
    kernel_name_ = base_operator->name();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::string kernel_type_{"Unknown"};
  static std::vector<KernelAttr> common_valid_types_with_bool_complex_;
  static std::vector<KernelAttr> common_two_valid_types_with_bool_complex_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESHAPE_CPU_KERNEL_H_
