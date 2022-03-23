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

#include "plugin/device/gpu/kernel/arrays/unsorted_segment_sum_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  UnsortedSegmentSumGpuKernelMod, double, int32_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  UnsortedSegmentSumGpuKernelMod, double, int64_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  UnsortedSegmentSumGpuKernelMod, float, int32_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  UnsortedSegmentSumGpuKernelMod, float, int64_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  UnsortedSegmentSumGpuKernelMod, half, int32_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  UnsortedSegmentSumGpuKernelMod, half, int64_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  UnsortedSegmentSumGpuKernelMod, int32_t, int32_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentSum,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  UnsortedSegmentSumGpuKernelMod, int, int64_t)
// Re-registration with 3 inputs - dynamic shape mode
// Int32
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentSumGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentSumGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentSumGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentSumGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentSumGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentSumGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentSumGpuKernelMod, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentSumGpuKernelMod, int32_t, int64_t)
// Int64
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentSumGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentSumGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentSumGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentSumGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentSumGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentSumGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentSumGpuKernelMod, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentSum,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentSumGpuKernelMod, int32_t, int64_t)
}  // namespace kernel
}  // namespace mindspore
