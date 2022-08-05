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

#include "plugin/device/gpu/kernel/arrays/strided_slice_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      StridedSliceGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      StridedSliceGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      StridedSliceGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      StridedSliceGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      StridedSliceGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      StridedSliceGpuKernelMod, short)  // NOLINT
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      StridedSliceGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      StridedSliceGpuKernelMod, uint64_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      StridedSliceGpuKernelMod, uint32_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      StridedSliceGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      StridedSliceGpuKernelMod, uchar)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      StridedSliceGpuKernelMod, bool)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      StridedSliceGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      StridedSliceGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      StridedSliceGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      StridedSliceGpuKernelMod, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      StridedSliceGpuKernelMod, int, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt16),
                      StridedSliceGpuKernelMod, short, int64_t)  // NOLINT
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt8),
                      StridedSliceGpuKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt64),
                      StridedSliceGpuKernelMod, uint64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt32),
                      StridedSliceGpuKernelMod, uint32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt16),
                      StridedSliceGpuKernelMod, uint16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt8),
                      StridedSliceGpuKernelMod, uchar, int64_t)
MS_REG_GPU_KERNEL_TWO(StridedSlice,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeBool),
                      StridedSliceGpuKernelMod, bool, int64_t)
}  // namespace kernel
}  // namespace mindspore
