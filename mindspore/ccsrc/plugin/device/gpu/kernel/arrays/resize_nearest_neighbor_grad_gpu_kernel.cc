/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/resize_nearest_neighbor_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(
  ResizeNearestNeighborGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  ResizeNearestNeighborGradGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  ResizeNearestNeighborGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  ResizeNearestNeighborGradGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  ResizeNearestNeighborGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  ResizeNearestNeighborGradGpuKernelMod, int)
MS_REG_GPU_KERNEL_TWO(
  ResizeNearestNeighborV2Grad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ResizeNearestNeighborGradGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ResizeNearestNeighborV2Grad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ResizeNearestNeighborGradGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ResizeNearestNeighborV2Grad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ResizeNearestNeighborGradGpuKernelMod, int, int32_t)
}  // namespace kernel
}  // namespace mindspore
