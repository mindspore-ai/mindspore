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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYNAMIC_STITCH_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYNAMIC_STITCH_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class DynamicStitchCPUKernel : public CPUKernel {
 public:
  DynamicStitchCPUKernel() = default;
  ~DynamicStitchCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 private:
  size_t input_tuple_num_{1};
};

MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, float);
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, int8_t)
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, int16_t)
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, int32_t)
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, int64_t)
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, uint8_t)
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, uint16_t)
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, uint32_t)
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, uint64_t)
MS_REG_CPU_KERNEL_T(DynamicStitch, KernelAttr(), DynamicStitchCPUKernel, bool)

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYNAMIC_STITCH_CPU_KERNEL_H_
