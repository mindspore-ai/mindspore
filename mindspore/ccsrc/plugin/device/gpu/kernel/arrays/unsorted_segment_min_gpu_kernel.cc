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

#include "plugin/device/gpu/kernel/arrays/unsorted_segment_min_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// Dynamic Mode - registered for int32/int64 3rd input
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMinGpuKernelMod, float)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMinGpuKernelMod, float)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMinGpuKernelMod, half)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMinGpuKernelMod, half)
// Int32
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMinGpuKernelMod, int)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMinGpuKernelMod, int)
// Int8
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt8),
                      UnsortedSegmentMinGpuKernelMod, int8_t)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt8),
                      UnsortedSegmentMinGpuKernelMod, int8_t)
// UInt8
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt8),
                      UnsortedSegmentMinGpuKernelMod, uint8_t)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt8),
                      UnsortedSegmentMinGpuKernelMod, uint8_t)
// Int16
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt16),
                      UnsortedSegmentMinGpuKernelMod, int16_t)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt16),
                      UnsortedSegmentMinGpuKernelMod, int16_t)
// UInt16
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt16),
                      UnsortedSegmentMinGpuKernelMod, uint16_t)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt16),
                      UnsortedSegmentMinGpuKernelMod, uint16_t)
// UInt32
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt32),
                      UnsortedSegmentMinGpuKernelMod, uint32_t)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt32),
                      UnsortedSegmentMinGpuKernelMod, uint32_t)
// Int64
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt64),
                      UnsortedSegmentMinGpuKernelMod, int64_t)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      UnsortedSegmentMinGpuKernelMod, int64_t)
// UInt64
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt64),
                      UnsortedSegmentMinGpuKernelMod, uint64_t)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt64),
                      UnsortedSegmentMinGpuKernelMod, uint64_t)
// Float64
MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentMinGpuKernelMod, double)

MS_REG_GPU_KERNEL_ONE(UnsortedSegmentMin,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentMinGpuKernelMod, double)
}  // namespace kernel
}  // namespace mindspore
