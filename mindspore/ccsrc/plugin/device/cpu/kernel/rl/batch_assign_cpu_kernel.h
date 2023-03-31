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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BATCH_ASSIGN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BATCH_ASSIGN_CPU_KERNEL_H_

#include <plugin/device/cpu/kernel/rl/batch_assign_cpu_base.h>
#include <string>
#include <vector>
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class BatchAssignCpuKernelMod : public BatchAssignCpuBaseMod {
 public:
  BatchAssignCpuKernelMod();
  ~BatchAssignCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  void InitKernel(const CNodePtr &kernel_node) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeBool)
                                                           .AddInputAttr(kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeFloat16)
                                                           .AddInputAttr(kNumberTypeFloat16)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeInt16)
                                                           .AddInputAttr(kNumberTypeInt16)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeInt8)
                                                           .AddInputAttr(kNumberTypeInt8)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeUInt32)
                                                           .AddInputAttr(kNumberTypeUInt32)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeUInt16)
                                                           .AddInputAttr(kNumberTypeUInt16)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeUInt64)
                                                           .AddInputAttr(kNumberTypeUInt64)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddAllSameAttr(true)
                                                           .AddInputAttr(kNumberTypeUInt8)
                                                           .AddInputAttr(kNumberTypeUInt8)
                                                           .AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  size_t elements_num_;
  bool lock_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BATCH_ASSIGN_CPU_KERNEL_H_
