/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/arrays/spacetobatch_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      SpaceToBatchGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      SpaceToBatchGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      SpaceToBatchGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      SpaceToBatchGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      SpaceToBatchGpuKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      SpaceToBatchGpuKernel, int8_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      SpaceToBatchGpuKernel, uint8_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      SpaceToBatchGpuKernel, uint16_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      SpaceToBatchGpuKernel, uint32_t)
MS_REG_GPU_KERNEL_ONE(SpaceToBatch, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      SpaceToBatchGpuKernel, uint64_t)
}  // namespace kernel
}  // namespace mindspore
