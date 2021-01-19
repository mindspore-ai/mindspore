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

#include "backend/kernel_compiler/gpu/nn/relu_grad_v2_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(
  ReluGradV2,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat64),
  ReluGradV2GpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  ReluGradV2,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat32),
  ReluGradV2GpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  ReluGradV2,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat16),
  ReluGradV2GpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  ReluGradV2, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt8),
  ReluGradV2GpuKernel, int8_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGradV2,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt16),
  ReluGradV2GpuKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGradV2,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
  ReluGradV2GpuKernel, int32_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGradV2,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
  ReluGradV2GpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGradV2,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt8),
  ReluGradV2GpuKernel, uint8_t)
}  // namespace kernel
}  // namespace mindspore
