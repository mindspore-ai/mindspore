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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROLLING_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROLLING_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace rolling {
enum Method : int {
  Max,
  Min,
  Mean,
  Sum,
  Std,
  Var,
};
}  // namespace rolling

class RollingCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  RollingCpuKernelMod() = default;
  ~RollingCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void InitInputOutputSize(const CNodePtr &kernel_node) override {
    DeprecatedNativeCpuKernelMod::InitInputOutputSize(kernel_node);
    func_obj_->InitInputOutputSize(kernel_node, &input_size_list_, &output_size_list_, &workspace_size_list_);
  }

 private:
  std::shared_ptr<DeprecatedCpuKernelFunc> func_obj_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SORT_CPU_KERNEL_H_
