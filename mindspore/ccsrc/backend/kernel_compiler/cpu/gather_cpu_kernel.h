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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/nnacl/base/gather_base.h"

namespace mindspore {
namespace kernel {
template <typename T>
class GatherV2CPUKernel : public CPUKernel {
 public:
  GatherV2CPUKernel() = default;
  ~GatherV2CPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void ParallelRun(const int8_t *input_addr, const int *indices_data, int8_t *output_addr, int thread_num);
  std::vector<size_t> input_shape_;
  std::vector<size_t> indices_shape_;
  std::vector<size_t> output_shape_;
  int64_t axis_{0};
  bool is_dynamic_shape_{false};
};

MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  GatherV2CPUKernel, uint8_t);
MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
  GatherV2CPUKernel, uint16_t);
MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherV2CPUKernel, uint32_t);
MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
  GatherV2CPUKernel, uint64_t);
MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  GatherV2CPUKernel, int8_t);
MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  GatherV2CPUKernel, int16_t);
MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherV2CPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  GatherV2CPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  GatherV2CPUKernel, float16);
MS_REG_CPU_KERNEL_T(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherV2CPUKernel, float);
MS_REG_CPU_KERNEL_T(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  GatherV2CPUKernel, double);
MS_REG_CPU_KERNEL_T(
  Gather, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  GatherV2CPUKernel, bool);
// dynamic shape ends
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeUInt8)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeUInt8),
                    GatherV2CPUKernel, uint8_t);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeUInt16)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeUInt16),
                    GatherV2CPUKernel, uint16_t);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeUInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeUInt32),
                    GatherV2CPUKernel, uint32_t);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeUInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeUInt64),
                    GatherV2CPUKernel, uint64_t);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt8)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt8),
                    GatherV2CPUKernel, int8_t);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt16)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt16),
                    GatherV2CPUKernel, int16_t);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt32),
                    GatherV2CPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt64),
                    GatherV2CPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat16),
                    GatherV2CPUKernel, float16);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat32),
                    GatherV2CPUKernel, float);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeFloat64),
                    GatherV2CPUKernel, double);
MS_REG_CPU_KERNEL_T(Gather,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeBool)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeBool),
                    GatherV2CPUKernel, bool);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHER_CPU_KERNEL_H_
