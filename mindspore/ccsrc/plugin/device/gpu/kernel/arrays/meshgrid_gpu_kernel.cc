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
#include "plugin/device/gpu/kernel/arrays/meshgrid_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Meshgrid,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      MeshgridGpuKernelMod, bool)

MS_REG_GPU_KERNEL_ONE(
  Meshgrid, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  MeshgridGpuKernelMod, half)

MS_REG_GPU_KERNEL_ONE(
  Meshgrid, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  MeshgridGpuKernelMod, float)

MS_REG_GPU_KERNEL_ONE(
  Meshgrid, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  MeshgridGpuKernelMod, double)

MS_REG_GPU_KERNEL_ONE(Meshgrid,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      MeshgridGpuKernelMod, uint8_t)

MS_REG_GPU_KERNEL_ONE(
  Meshgrid, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
  MeshgridGpuKernelMod, uint16_t)

MS_REG_GPU_KERNEL_ONE(
  Meshgrid, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  MeshgridGpuKernelMod, uint32_t)

MS_REG_GPU_KERNEL_ONE(
  Meshgrid, KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
  MeshgridGpuKernelMod, uint64_t)

MS_REG_GPU_KERNEL_ONE(Meshgrid,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      MeshgridGpuKernelMod, int8_t)

MS_REG_GPU_KERNEL_ONE(Meshgrid,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      MeshgridGpuKernelMod, int16_t)

MS_REG_GPU_KERNEL_ONE(Meshgrid,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      MeshgridGpuKernelMod, int32_t)

MS_REG_GPU_KERNEL_ONE(Meshgrid,
                      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      MeshgridGpuKernelMod, int64_t)
}  // namespace kernel
}  // namespace mindspore
