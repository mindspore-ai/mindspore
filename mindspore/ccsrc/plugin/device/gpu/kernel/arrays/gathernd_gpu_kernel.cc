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

#include "plugin/device/gpu/kernel/arrays/gathernd_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  GatherNdFwdGpuKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherNdFwdGpuKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  GatherNdFwdGpuKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherNdFwdGpuKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  GatherNdFwdGpuKernelMod, short, int)  // NOLINT
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
  GatherNdFwdGpuKernelMod, uint, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  GatherNdFwdGpuKernelMod, char, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  GatherNdFwdGpuKernelMod, uchar, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  GatherNdFwdGpuKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  GatherNdFwdGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  GatherNdFwdGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  GatherNdFwdGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  GatherNdFwdGpuKernelMod, int, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  GatherNdFwdGpuKernelMod, short, int64_t)  // NOLINT
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
  GatherNdFwdGpuKernelMod, uint, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
  GatherNdFwdGpuKernelMod, char, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
  GatherNdFwdGpuKernelMod, uchar, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  GatherNdFwdGpuKernelMod, bool, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
  GatherNdFwdGpuKernelMod, cuComplex, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
  GatherNdFwdGpuKernelMod, cuDoubleComplex, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
  GatherNdFwdGpuKernelMod, cuComplex, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
  GatherNdFwdGpuKernelMod, cuDoubleComplex, int64_t)
}  // namespace kernel
}  // namespace mindspore
