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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_D_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename I>
class GatherDCPUKernel : public CPUKernel {
 public:
  GatherDCPUKernel() = default;
  ~GatherDCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> index_shape_;
  std::vector<size_t> output_shape_;
};

MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, float, int32_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, float, int64_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, float16, int32_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, float16, int64_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, int32_t, int32_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, int32_t, int64_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, int64_t, int32_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, int64_t, int64_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, bool, int32_t);
MS_REG_CPU_KERNEL_T_S(GatherD, KernelAttr(), GatherDCPUKernel, bool, int64_t);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_D_CPU_KERNEL_H_
