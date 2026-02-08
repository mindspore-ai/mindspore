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

#include "kernel/gpu/cuda/nn/rmsprop_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      RMSPropGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      RMSPropGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      RMSPropGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      RMSPropGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddOutputAttr(kNumberTypeInt16),
                      RMSPropGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      RMSPropGpuKernelMod, int32_t)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      RMSPropGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      RMSPropGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddOutputAttr(kNumberTypeUInt16),
                      RMSPropGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddOutputAttr(kNumberTypeUInt32),
                      RMSPropGpuKernelMod, uint32_t)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddOutputAttr(kNumberTypeUInt64),
                      RMSPropGpuKernelMod, uint64_t)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddOutputAttr(kNumberTypeComplex64),
                      RMSPropGpuKernelMod, utils::Complex<float>)
MS_REG_GPU_KERNEL_ONE(ApplyRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddOutputAttr(kNumberTypeComplex128),
                      RMSPropGpuKernelMod, utils::Complex<double>)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      RMSPropGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      RMSPropGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      RMSPropGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      RMSPropGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddOutputAttr(kNumberTypeInt16),
                      RMSPropGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      RMSPropGpuKernelMod, int32_t)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      RMSPropGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      RMSPropGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddOutputAttr(kNumberTypeUInt16),
                      RMSPropGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddOutputAttr(kNumberTypeUInt32),
                      RMSPropGpuKernelMod, uint32_t)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddOutputAttr(kNumberTypeUInt64),
                      RMSPropGpuKernelMod, uint64_t)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddOutputAttr(kNumberTypeComplex64),
                      RMSPropGpuKernelMod, utils::Complex<float>)
MS_REG_GPU_KERNEL_ONE(ApplyCenteredRMSProp,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddOutputAttr(kNumberTypeComplex128),
                      RMSPropGpuKernelMod, utils::Complex<double>)
}  // namespace kernel
}  // namespace mindspore
