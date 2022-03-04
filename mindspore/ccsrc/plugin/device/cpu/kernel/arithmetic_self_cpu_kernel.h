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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_

#include <complex>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnknown = "Unknown";

class ArithmeticSelfCpuKernelMod : public NativeCpuKernelMod {
 public:
  ArithmeticSelfCpuKernelMod() = default;
  explicit ArithmeticSelfCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ArithmeticSelfCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
  std::string kernel_type_{kUnknown};
};

using LaunchFunc = std::function<bool(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
class IdentityCpuKernelMod : public NativeCpuKernelMod {
 public:
  IdentityCpuKernelMod() = default;
  ~IdentityCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), 1, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), 1, kernel_name_);
    return kernel_func_(inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  LaunchFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_
