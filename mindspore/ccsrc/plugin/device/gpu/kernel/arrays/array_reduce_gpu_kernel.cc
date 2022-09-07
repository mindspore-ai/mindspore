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

#include "plugin/device/gpu/kernel/arrays/array_reduce_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(ReduceMax, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      ArrayReduceGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ReduceMax, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ArrayReduceGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ReduceMax, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ArrayReduceGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ReduceMean, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      ArrayReduceGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ReduceMean, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ArrayReduceGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ReduceMean, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ArrayReduceGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      ArrayReduceGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ArrayReduceGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ArrayReduceGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      ArrayReduceGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(ReduceMin, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      ArrayReduceGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ReduceMin, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ArrayReduceGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ReduceMin, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ArrayReduceGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ReduceAny, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      ArrayReduceGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(ReduceAll, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      ArrayReduceGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(ReduceProd, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      ArrayReduceGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(ReduceProd, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ArrayReduceGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ReduceProd, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ArrayReduceGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ReduceProd, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      ArrayReduceGpuKernelMod, double)

// dynamic
MS_REG_GPU_KERNEL_TWO(
  ReduceMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMean,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMean,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMean,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMean,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMean,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMean,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMin,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMin,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMin,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMin,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMin,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceMin,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceAll, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArrayReduceGpuKernelMod, bool, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceAll, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArrayReduceGpuKernelMod, bool, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceAny, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArrayReduceGpuKernelMod, bool, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceAny, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArrayReduceGpuKernelMod, bool, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceProd, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
  ArrayReduceGpuKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceProd,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceProd,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceProd,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceProd, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  ArrayReduceGpuKernelMod, int8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceProd,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceProd,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceProd,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceSum, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArrayReduceGpuKernelMod, bool, int64_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArrayReduceGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArrayReduceGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceSum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArrayReduceGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(
  ReduceSum, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArrayReduceGpuKernelMod, bool, int32_t)
}  // namespace kernel
}  // namespace mindspore
