/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
                      FloatStatusGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(FloatStatus, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
                      FloatStatusGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(IsInf, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(IsInf, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(IsNan, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(IsNan, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(IsFinite, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(IsFinite, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
                      FloatStatusGpuKernel, half)
}  // namespace kernel
}  // namespace mindspore
