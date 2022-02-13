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
#include "plugin/device/gpu/kernel/random/randperm_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Randperm, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
                      RandpermGpuKernelMod, int8_t)

MS_REG_GPU_KERNEL_ONE(Randperm, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
                      RandpermGpuKernelMod, int16_t)

MS_REG_GPU_KERNEL_ONE(Randperm, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      RandpermGpuKernelMod, int32_t)

MS_REG_GPU_KERNEL_ONE(Randperm, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
                      RandpermGpuKernelMod, int64_t)

MS_REG_GPU_KERNEL_ONE(Randperm, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
                      RandpermGpuKernelMod, uint8_t)

MS_REG_GPU_KERNEL_ONE(Randperm, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
                      RandpermGpuKernelMod, uint16_t)

MS_REG_GPU_KERNEL_ONE(Randperm, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
                      RandpermGpuKernelMod, uint32_t)

MS_REG_GPU_KERNEL_ONE(Randperm, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
                      RandpermGpuKernelMod, uint64_t)
}  // namespace kernel
}  // namespace mindspore
