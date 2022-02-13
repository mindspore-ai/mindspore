/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_COALESCE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_COALESCE_CPU_KERNEL_H_

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class CoalesceCpuKernelMod : public NativeCpuKernelMod {
 public:
  CoalesceCpuKernelMod() = default;
  ~CoalesceCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  void CheckParam(const CNodePtr &kernel_node);
  void Check(const std::vector<AddressPtr> &inputs);
  TypeId dtype_{kTypeUnknown};
  size_t values_size_{0};
  size_t shape_size_{0};
  size_t jump = 0;
  CNodeWeakPtr node_wpt_;
};

MS_REG_CPU_KERNEL(Coalesce,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeInt64),
                  CoalesceCpuKernelMod);

MS_REG_CPU_KERNEL(Coalesce,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeInt64),
                  CoalesceCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_COALESCE_CPU_KERNEL_H_
