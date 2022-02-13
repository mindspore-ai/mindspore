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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <thread>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"
#include "plugin/device/cpu/kernel/nnacl/base/split_base.h"

namespace mindspore {
namespace kernel {
template <typename T>
class SplitCpuKernelMod : public NativeCpuKernelMod {
 public:
  SplitCpuKernelMod() = default;
  ~SplitCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);

  void LaunchSplit(T *input, T **output, size_t size);

  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  void InitInputOutputSize(const CNodePtr &kernel_node) override;

  int64_t axis_{0};
  size_t output_num_{1};
  std::vector<int> input_shape_;
};

MS_REG_CPU_KERNEL_T(Split, KernelAttr(), SplitCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(Split, KernelAttr(), SplitCpuKernelMod, float16);
MS_REG_CPU_KERNEL_T(Split, KernelAttr(), SplitCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(Split, KernelAttr(), SplitCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(Split, KernelAttr(), SplitCpuKernelMod, uint32_t);
MS_REG_CPU_KERNEL_T(Split, KernelAttr(), SplitCpuKernelMod, int64_t);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPLIT_CPU_KERNEL_H_
