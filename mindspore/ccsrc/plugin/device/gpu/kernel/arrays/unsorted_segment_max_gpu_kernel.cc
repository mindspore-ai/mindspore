/** copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/unsorted_segment_max_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// Dynamic Mode - registered for int32/int64 3rd input
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMaxGpuKernelMod, float, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMaxGpuKernelMod, float, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMaxGpuKernelMod, float, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMaxGpuKernelMod, float, int64_t)
// Float32
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMaxGpuKernelMod, half, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMaxGpuKernelMod, half, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMaxGpuKernelMod, half, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMaxGpuKernelMod, half, int64_t)
// Int32
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMaxGpuKernelMod, int, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMaxGpuKernelMod, int, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMaxGpuKernelMod, int, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMaxGpuKernelMod, int, int64_t)
// Int8
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt8),
                      UnsortedSegmentMaxGpuKernelMod, int8_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt8),
                      UnsortedSegmentMaxGpuKernelMod, int8_t, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt8),
                      UnsortedSegmentMaxGpuKernelMod, int8_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt8),
                      UnsortedSegmentMaxGpuKernelMod, int8_t, int64_t)
// UInt8
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt8),
                      UnsortedSegmentMaxGpuKernelMod, uint8_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt8),
                      UnsortedSegmentMaxGpuKernelMod, uint8_t, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt8),
                      UnsortedSegmentMaxGpuKernelMod, uint8_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt8),
                      UnsortedSegmentMaxGpuKernelMod, uint8_t, int64_t)
// Int16
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt16),
                      UnsortedSegmentMaxGpuKernelMod, int16_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt16),
                      UnsortedSegmentMaxGpuKernelMod, int16_t, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt16),
                      UnsortedSegmentMaxGpuKernelMod, int16_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt16),
                      UnsortedSegmentMaxGpuKernelMod, int16_t, int64_t)
// UInt16
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt16),
                      UnsortedSegmentMaxGpuKernelMod, uint16_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt16),
                      UnsortedSegmentMaxGpuKernelMod, uint16_t, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt16),
                      UnsortedSegmentMaxGpuKernelMod, uint16_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt16),
                      UnsortedSegmentMaxGpuKernelMod, uint16_t, int64_t)
// UInt32
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt32),
                      UnsortedSegmentMaxGpuKernelMod, uint32_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt32),
                      UnsortedSegmentMaxGpuKernelMod, uint32_t, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt32),
                      UnsortedSegmentMaxGpuKernelMod, uint32_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt32),
                      UnsortedSegmentMaxGpuKernelMod, uint32_t, int64_t)
// Int64
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt64),
                      UnsortedSegmentMaxGpuKernelMod, int64_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt64),
                      UnsortedSegmentMaxGpuKernelMod, int64_t, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      UnsortedSegmentMaxGpuKernelMod, int64_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      UnsortedSegmentMaxGpuKernelMod, int64_t, int64_t)
// UInt64
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt64),
                      UnsortedSegmentMaxGpuKernelMod, uint64_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt64),
                      UnsortedSegmentMaxGpuKernelMod, uint64_t, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt64),
                      UnsortedSegmentMaxGpuKernelMod, uint64_t, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt64),
                      UnsortedSegmentMaxGpuKernelMod, uint64_t, int64_t)
// Float64
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentMaxGpuKernelMod, double, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentMaxGpuKernelMod, double, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentMaxGpuKernelMod, double, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      UnsortedSegmentMaxGpuKernelMod, double, int64_t)
}  // namespace kernel
}  // namespace mindspore
