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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_CPU_KERNEL_H_

#include <vector>
#include <unordered_map>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename S, typename T>
struct ComputeParams {
  T *target_{nullptr};
  S *indices_{nullptr};
  T *updates_{nullptr};
  int unit_size_{0};
  int indices_unit_rank_{0};
  std::vector<int> *out_strides_{nullptr};
  size_t target_mem_size_{0};
};

template <typename S, typename T>
class ScatterNdCPUKernel : public CPUKernel {
 public:
  ScatterNdCPUKernel() = default;
  ~ScatterNdCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void Check(const CNodePtr &kernel_node);

  int unit_size_{1};
  size_t num_units_{1};
  int indices_unit_rank_{0};
  std::vector<int> out_strides_;
};

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ScatterNdCPUKernel, int64_t, double);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ScatterNdCPUKernel, int64_t, float);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ScatterNdCPUKernel, int64_t, int64_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ScatterNdCPUKernel, int64_t, int32_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  ScatterNdCPUKernel, int64_t, int16_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  ScatterNdCPUKernel, int64_t, int8_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
  ScatterNdCPUKernel, int64_t, uint64_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  ScatterNdCPUKernel, int64_t, uint32_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
  ScatterNdCPUKernel, int64_t, uint16_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  ScatterNdCPUKernel, int64_t, uint8_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ScatterNdCPUKernel, int32_t, double);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ScatterNdCPUKernel, int32_t, float);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ScatterNdCPUKernel, int32_t, int64_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ScatterNdCPUKernel, int32_t, int32_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  ScatterNdCPUKernel, int32_t, int16_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  ScatterNdCPUKernel, int32_t, int8_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
  ScatterNdCPUKernel, int32_t, uint64_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  ScatterNdCPUKernel, int32_t, uint32_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
  ScatterNdCPUKernel, int32_t, uint16_t);

MS_REG_CPU_KERNEL_T_S(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  ScatterNdCPUKernel, int32_t, uint8_t);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_CPU_KERNEL_H_
