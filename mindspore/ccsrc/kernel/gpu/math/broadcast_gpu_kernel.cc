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

#include "kernel/gpu/math/broadcast_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// fp32
MS_REG_GPU_KERNEL_TWO(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, float, bool)
MS_REG_GPU_KERNEL_TWO(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, float, bool)
MS_REG_GPU_KERNEL_TWO(
  Maximum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float, float)

// fp16
MS_REG_GPU_KERNEL_TWO(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, half, bool)
MS_REG_GPU_KERNEL_TWO(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, half, bool)
MS_REG_GPU_KERNEL_TWO(
  Maximum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half, half)
}  // namespace kernel
}  // namespace mindspore
