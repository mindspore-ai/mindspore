/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/arrays/squeeze_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      SqueezeGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      SqueezeGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      SqueezeGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      SqueezeGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      SqueezeGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      SqueezeGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      SqueezeGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      SqueezeGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      SqueezeGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      SqueezeGpuKernelMod, uint32_t)
MS_REG_GPU_KERNEL_ONE(Squeeze, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      SqueezeGpuKernelMod, bool)
}  // namespace kernel
}  // namespace mindspore
