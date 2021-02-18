/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/arrays/zeroslike_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      ZerosLikeGpuKernel, bool)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      ZerosLikeGpuKernel, int8_t)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      ZerosLikeGpuKernel, uint8_t)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      ZerosLikeGpuKernel, int32_t)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ZerosLikeGpuKernel, half)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ZerosLikeGpuKernel, float)

MS_REG_GPU_KERNEL_ONE(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      ZerosLikeGpuKernel, double)
}  // namespace kernel
}  // namespace mindspore
