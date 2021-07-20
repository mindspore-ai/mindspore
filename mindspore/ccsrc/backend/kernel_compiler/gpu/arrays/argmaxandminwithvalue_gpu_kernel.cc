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

#include "backend/kernel_compiler/gpu/arrays/argmaxandminwithvalue_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArgMaxAndMinWithValueGpuKernel, double, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArgMaxAndMinWithValueGpuKernel, float, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMaxWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArgMaxAndMinWithValueGpuKernel, half, int)

MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ArgMaxAndMinWithValueGpuKernel, double, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ArgMaxAndMinWithValueGpuKernel, float, int)
MS_REG_GPU_KERNEL_TWO(
  ArgMinWithValue,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  ArgMaxAndMinWithValueGpuKernel, half, int)
}  // namespace kernel
}  // namespace mindspore
