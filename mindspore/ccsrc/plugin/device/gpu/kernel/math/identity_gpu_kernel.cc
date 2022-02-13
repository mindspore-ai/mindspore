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

#include "plugin/device/gpu/kernel/math/identity_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      IdentityGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      IdentityGpuKernelMod, float);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      IdentityGpuKernelMod, half);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      IdentityGpuKernelMod, uint64_t);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      IdentityGpuKernelMod, int64_t);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      IdentityGpuKernelMod, uint32_t);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      IdentityGpuKernelMod, int32_t);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      IdentityGpuKernelMod, uint16_t);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      IdentityGpuKernelMod, int16_t);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      IdentityGpuKernelMod, uint8_t);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      IdentityGpuKernelMod, int8_t);
MS_REG_GPU_KERNEL_ONE(Identity, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      IdentityGpuKernelMod, bool);
}  // namespace kernel
}  // namespace mindspore
