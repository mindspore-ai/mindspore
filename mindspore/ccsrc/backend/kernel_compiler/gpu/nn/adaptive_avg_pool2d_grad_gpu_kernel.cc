/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/nn/adaptive_avg_pool2d_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(
  AdaptiveAvgPool2DGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  AdaptiveAvgPool2DGradKernel, half)
MS_REG_GPU_KERNEL_ONE(
  AdaptiveAvgPool2DGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  AdaptiveAvgPool2DGradKernel, float)
MS_REG_GPU_KERNEL_ONE(
  AdaptiveAvgPool2DGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  AdaptiveAvgPool2DGradKernel, double)
}  // namespace kernel
}  // namespace mindspore
