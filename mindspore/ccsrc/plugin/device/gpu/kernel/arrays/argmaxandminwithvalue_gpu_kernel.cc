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

#include "plugin/device/gpu/kernel/arrays/argmaxandminwithvalue_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  ArgMaxAndMinWithValueGpuKernelMod, int8_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  ArgMaxAndMinWithValueGpuKernelMod, int64_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  ArgMaxAndMinWithValueGpuKernelMod, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
  ArgMaxAndMinWithValueGpuKernelMod, uint64_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  ArgMaxAndMinWithValueGpuKernelMod, int16_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArgMaxAndMinWithValueGpuKernelMod, int32_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
  ArgMaxAndMinWithValueGpuKernelMod, uint16_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
  ArgMaxAndMinWithValueGpuKernelMod, uint32_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArgMaxAndMinWithValueGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArgMaxAndMinWithValueGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArgMaxAndMinWithValueGpuKernelMod, half, int)

MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  ArgMaxAndMinWithValueGpuKernelMod, int8_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  ArgMaxAndMinWithValueGpuKernelMod, int64_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  ArgMaxAndMinWithValueGpuKernelMod, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
  ArgMaxAndMinWithValueGpuKernelMod, uint64_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  ArgMaxAndMinWithValueGpuKernelMod, int16_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArgMaxAndMinWithValueGpuKernelMod, int32_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
  ArgMaxAndMinWithValueGpuKernelMod, uint16_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
  ArgMaxAndMinWithValueGpuKernelMod, uint32_t, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArgMaxAndMinWithValueGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArgMaxAndMinWithValueGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArgMaxAndMinWithValueGpuKernelMod, half, int)
}  // namespace kernel
}  // namespace mindspore
