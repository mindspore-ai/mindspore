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

#include "backend/kernel_compiler/gpu/math/float_status_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(FloatStatus, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      FloatStatusGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(FloatStatus, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
                      FloatStatusGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(FloatStatus, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
                      FloatStatusGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(IsInf, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(IsInf, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(IsInf, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(IsNan, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(IsNan, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(IsNan, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(IsFinite, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(IsFinite, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(IsFinite, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernelMod, double)
}  // namespace kernel
}  // namespace mindspore
