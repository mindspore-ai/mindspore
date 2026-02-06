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

#include "kernel/gpu/cuda/arrays/pack_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Stack,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      PackFwdGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(Stack,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      PackFwdGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(Stack,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      PackFwdGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Stack,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      PackFwdGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(Stack,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      PackFwdGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(Stack,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      PackFwdGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(
  Stack, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
  PackFwdGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(
  Stack, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  PackFwdGpuKernelMod, uint32_t)
MS_REG_GPU_KERNEL_ONE(
  Stack, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
  PackFwdGpuKernelMod, uint64_t)
MS_REG_GPU_KERNEL_ONE(
  Stack, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  PackFwdGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Stack, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  PackFwdGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Stack, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  PackFwdGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Stack, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
  PackFwdGpuKernelMod, utils::Complex<float>)
MS_REG_GPU_KERNEL_ONE(
  Stack, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
  PackFwdGpuKernelMod, utils::Complex<double>)
}  // namespace kernel
}  // namespace mindspore
