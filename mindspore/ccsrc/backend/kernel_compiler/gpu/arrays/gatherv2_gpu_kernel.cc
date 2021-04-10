/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/arrays/gatherv2_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  GatherV2GpuFwdKernel, double, int)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  GatherV2GpuFwdKernel, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherV2GpuFwdKernel, float, int)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  GatherV2GpuFwdKernel, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  GatherV2GpuFwdKernel, half, int)
MS_REG_GPU_KERNEL_TWO(
  Gather,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  GatherV2GpuFwdKernel, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherV2GpuFwdKernel, int, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  GatherV2GpuFwdKernel, int, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  GatherV2GpuFwdKernel, int16_t, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  GatherV2GpuFwdKernel, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  GatherV2GpuFwdKernel, int8_t, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
  GatherV2GpuFwdKernel, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  GatherV2GpuFwdKernel, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(
  Gather, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
  GatherV2GpuFwdKernel, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      GatherV2GpuFwdKernel, float, int)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      GatherV2GpuFwdKernel, float, int64_t)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      GatherV2GpuFwdKernel, half, int)
MS_REG_GPU_KERNEL_TWO(Gather,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      GatherV2GpuFwdKernel, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  SparseGatherV2,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherV2GpuFwdKernel, float, int)
MS_REG_GPU_KERNEL_TWO(
  SparseGatherV2,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  GatherV2GpuFwdKernel, half, int)
MS_REG_GPU_KERNEL_TWO(SparseGatherV2,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      GatherV2GpuFwdKernel, float, int)
MS_REG_GPU_KERNEL_TWO(SparseGatherV2,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      GatherV2GpuFwdKernel, half, int)
}  // namespace kernel
}  // namespace mindspore
