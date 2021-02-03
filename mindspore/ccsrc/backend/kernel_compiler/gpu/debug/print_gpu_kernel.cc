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

#include "backend/kernel_compiler/gpu/debug/print_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, bool)
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, int8_t)
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, uint8_t)
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, uint16_t)
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, uint32_t)
MS_REG_GPU_KERNEL_ONE(Print,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
                      PrintGpuKernel, uint64_t)
MS_REG_GPU_KERNEL_ONE(
  Print, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
  PrintGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Print, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
  PrintGpuKernel, float)
}  // namespace kernel
}  // namespace mindspore
