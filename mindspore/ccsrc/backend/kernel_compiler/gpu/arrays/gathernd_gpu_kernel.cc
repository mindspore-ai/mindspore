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

#include "backend/kernel_compiler/gpu/arrays/gathernd_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  GatherNdGpuFwdKernel, double, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherNdGpuFwdKernel, float, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  GatherNdGpuFwdKernel, half, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherNdGpuFwdKernel, int, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  GatherNdGpuFwdKernel, short, int)  // NOLINT
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  GatherNdGpuFwdKernel, uchar, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  GatherNdGpuFwdKernel, bool, int)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  GatherNdGpuFwdKernel, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  GatherNdGpuFwdKernel, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  GatherNdGpuFwdKernel, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  GatherNdGpuFwdKernel, int, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  GatherNdGpuFwdKernel, short, int64_t)  // NOLINT
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
  GatherNdGpuFwdKernel, uchar, int64_t)
MS_REG_GPU_KERNEL_TWO(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  GatherNdGpuFwdKernel, bool, int64_t)
}  // namespace kernel
}  // namespace mindspore
