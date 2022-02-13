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
#include <cstdint>

#include "plugin/device/gpu/kernel/arrays/zeroslike_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      ZerosLikeGpuKernelMod, bool)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      ZerosLikeGpuKernelMod, int8_t)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      ZerosLikeGpuKernelMod, uint8_t)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      ZerosLikeGpuKernelMod, int32_t)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ZerosLikeGpuKernelMod, half)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ZerosLikeGpuKernelMod, float)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      ZerosLikeGpuKernelMod, double)
}  // namespace kernel
}  // namespace mindspore
