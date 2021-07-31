/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/arrays/gather_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  GatherGradGpuKernel, int, double)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  GatherGradGpuKernel, int64_t, double)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  GatherGradGpuKernel, int, float)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  GatherGradGpuKernel, int64_t, float)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  GatherGradGpuKernel, int, half)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  GatherGradGpuKernel, int64_t, half)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherGradGpuKernel, int, int)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherGradGpuKernel, int64_t, int)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  GatherGradGpuKernel, int, int8_t)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  GatherGradGpuKernel, int64_t, int8_t)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  GatherGradGpuKernel, int, int16_t)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  GatherGradGpuKernel, int64_t, int16_t)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  GatherGradGpuKernel, int, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  GatherGradGpuKernel, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  GatherGradGpuKernel, int, uchar)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  GatherGradGpuKernel, int64_t, uchar)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherGradGpuKernel, int, uint)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherGradGpuKernel, int64_t, uint)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  GatherGradGpuKernel, int, bool)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  GatherGradGpuKernel, int64_t, bool)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherGradGpuKernel, int, uint32_t)
MS_REG_GPU_KERNEL_TWO(
  GatherDGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherGradGpuKernel, int64_t, uint32_t)
}  // namespace kernel
}  // namespace mindspore
