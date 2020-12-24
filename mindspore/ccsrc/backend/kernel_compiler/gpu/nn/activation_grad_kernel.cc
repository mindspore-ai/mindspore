/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/nn/activation_grad_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(
  ReluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ActivationGradGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ActivationGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ActivationGradGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ActivationGradGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ActivationGradGpuKernel, int32_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  ActivationGradGpuKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  ActivationGradGpuKernel, int8_t)
MS_REG_GPU_KERNEL_ONE(
  ReluGrad, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  ActivationGradGpuKernel, uint8_t)

MS_REG_GPU_KERNEL_ONE(
  ReLU6Grad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ActivationGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  ReLU6Grad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ActivationGradGpuKernel, half)

MS_REG_GPU_KERNEL_ONE(
  TanhGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ActivationGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  TanhGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ActivationGradGpuKernel, half)

MS_REG_GPU_KERNEL_ONE(
  EluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ActivationGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  EluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ActivationGradGpuKernel, half)

MS_REG_GPU_KERNEL_ONE(
  SigmoidGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ActivationGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  SigmoidGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ActivationGradGpuKernel, half)
}  // namespace kernel
}  // namespace mindspore
