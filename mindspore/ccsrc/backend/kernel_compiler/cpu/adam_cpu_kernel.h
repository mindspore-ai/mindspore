/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAM_CPU_KERNEL_H_

#include <vector>
#include <memory>

#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class AdamCPUKernel : public CPUKernel {
 public:
  AdamCPUKernel() = default;
  ~AdamCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchAdam(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  void LaunchAdamNnacl(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  bool use_nesterov_{false};
  TypeId dtype_{kTypeUnknown};
  enum input_list_ { VAR, M, V, BETA1_POWER, BETA2_POWER, LR, BETA1, BETA2, EPSILON, GRAD };
};

MS_REG_CPU_KERNEL(Adam, KernelAttr(), AdamCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAM_CPU_KERNEL_H_
