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

#include "plugin/device/gpu/kernel/arrays/tile_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Tile, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      TileGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Tile, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      TileGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Tile, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      TileGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Tile, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      TileGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(Tile, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      TileGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Tile, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      TileGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(Tile, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      TileGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Tile, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), TileGpuKernelMod,
                      bool)
}  // namespace kernel
}  // namespace mindspore
