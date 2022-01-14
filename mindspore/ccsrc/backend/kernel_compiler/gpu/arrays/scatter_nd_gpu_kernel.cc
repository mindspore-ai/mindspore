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

#include "backend/kernel_compiler/gpu/arrays/scatter_nd_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ScatterNdFwdGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ScatterNdFwdGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ScatterNdFwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ScatterNdFwdGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ScatterNdFwdGpuKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ScatterNdFwdGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ScatterNdFwdGpuKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ScatterNdFwdGpuKernelMod, int, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  ScatterNdFwdGpuKernelMod, short, int)  // NOLINT
MS_REG_GPU_KERNEL_TWO(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  ScatterNdFwdGpuKernelMod, short, int64_t)  // NOLINT
MS_REG_GPU_KERNEL_TWO(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  ScatterNdFwdGpuKernelMod, uchar, int)
MS_REG_GPU_KERNEL_TWO(
  ScatterNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  ScatterNdFwdGpuKernelMod, uchar, int64_t)
}  // namespace kernel
}  // namespace mindspore
