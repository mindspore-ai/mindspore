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
#include "kernel/gpu/cuda/other/gpu_convert_to_dynamic_shape_gpu_kernel.h"

#include <cstdint>

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      GpuConvertToDynamicShapeGpuKernelMod, bool)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      GpuConvertToDynamicShapeGpuKernelMod, half)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      GpuConvertToDynamicShapeGpuKernelMod, float)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      GpuConvertToDynamicShapeGpuKernelMod, double)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      GpuConvertToDynamicShapeGpuKernelMod, int8_t)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      GpuConvertToDynamicShapeGpuKernelMod, int16_t)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      GpuConvertToDynamicShapeGpuKernelMod, int32_t)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      GpuConvertToDynamicShapeGpuKernelMod, int64_t)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      GpuConvertToDynamicShapeGpuKernelMod, uint8_t)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      GpuConvertToDynamicShapeGpuKernelMod, uint16_t)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      GpuConvertToDynamicShapeGpuKernelMod, uint32_t)

MS_REG_GPU_KERNEL_ONE(GpuConvertToDynamicShape,
                      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      GpuConvertToDynamicShapeGpuKernelMod, uint64_t)
}  // namespace kernel
}  // namespace mindspore
