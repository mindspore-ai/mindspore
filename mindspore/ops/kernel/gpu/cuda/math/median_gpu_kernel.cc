/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/math/median_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  Median, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, uint16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, uint32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, int32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, uint64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  Median,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
  MedianGpuKernelMod, double, int64_t)
}  // namespace kernel
}  // namespace mindspore
