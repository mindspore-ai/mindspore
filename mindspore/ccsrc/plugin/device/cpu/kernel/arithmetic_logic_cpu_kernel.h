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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_LOGIC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_LOGIC_CPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <limits>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ArithmeticLogicCpuKernelMod : public NativeCpuKernelMod {
 public:
  ArithmeticLogicCpuKernelMod() = default;
  explicit ArithmeticLogicCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ArithmeticLogicCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    const size_t kInputsNum = 2;
    const size_t kOutputsNum = 1;
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
  std::string kernel_type_{"Unknown"};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_LOGIC_CPU_KERNEL_H_
