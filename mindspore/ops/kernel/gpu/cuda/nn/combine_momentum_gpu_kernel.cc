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

#include "kernel/gpu/cuda/nn/combine_momentum_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_THREE(CombineMomentum,
                        KernelAttr()
                          .AddAllSameAttr(true, 5, true)
                          .AddInputAttr(kNumberTypeFloat32)  // variable
                          .AddInputAttr(kNumberTypeFloat32)  // accumulation
                          .AddInputAttr(kNumberTypeFloat32)  // learning_rate
                          .AddInputAttr(kNumberTypeFloat32)  // gradient
                          .AddInputAttr(kNumberTypeFloat32)  // momentum
                          .AddOutputAttr(kNumberTypeFloat32),
                        CombineMomentumGpuKernelMod, float, float, float)
MS_REG_GPU_KERNEL_THREE(CombineMomentum,
                        KernelAttr()
                          .AddAllSameAttr(true, 5, true)
                          .AddInputAttr(kNumberTypeFloat16)  // variable
                          .AddInputAttr(kNumberTypeFloat16)  // accumulation
                          .AddInputAttr(kNumberTypeFloat16)  // learning_rate
                          .AddInputAttr(kNumberTypeFloat16)  // gradient
                          .AddInputAttr(kNumberTypeFloat16)  // momentum
                          .AddOutputAttr(kNumberTypeFloat16),
                        CombineMomentumGpuKernelMod, half, half, half)
MS_REG_GPU_KERNEL_THREE(CombineMomentum,
                        KernelAttr()
                          .AddAllSameAttr(true, 5, true)
                          .AddInputAttr(kNumberTypeFloat16)  // variable
                          .AddInputAttr(kNumberTypeFloat16)  // accumulation
                          .AddInputAttr(kNumberTypeFloat32)  // learning_rate
                          .AddInputAttr(kNumberTypeFloat16)  // gradient
                          .AddInputAttr(kNumberTypeFloat32)  // momentum
                          .AddOutputAttr(kNumberTypeFloat16),
                        CombineMomentumGpuKernelMod, half, float, half)
MS_REG_GPU_KERNEL_THREE(CombineMomentum,
                        KernelAttr()
                          .AddAllSameAttr(true, 5, true)
                          .AddInputAttr(kNumberTypeFloat32)  // variable
                          .AddInputAttr(kNumberTypeFloat32)  // accumulation
                          .AddInputAttr(kNumberTypeFloat32)  // learning_rate
                          .AddInputAttr(kNumberTypeFloat16)  // gradient
                          .AddInputAttr(kNumberTypeFloat32)  // momentum
                          .AddOutputAttr(kNumberTypeFloat32),
                        CombineMomentumGpuKernelMod, float, float, half)
MS_REG_GPU_KERNEL_THREE(CombineScaleMomentum,
                        KernelAttr()
                          .AddAllSameAttr(true, 6, true)
                          .AddInputAttr(kNumberTypeFloat32)  // scale
                          .AddInputAttr(kNumberTypeFloat32)  // variable
                          .AddInputAttr(kNumberTypeFloat32)  // accumulation
                          .AddInputAttr(kNumberTypeFloat32)  // learning_rate
                          .AddInputAttr(kNumberTypeFloat32)  // gradient
                          .AddInputAttr(kNumberTypeFloat32)  // momentum
                          .AddOutputAttr(kNumberTypeFloat32),
                        CombineMomentumGpuKernelMod, float, float, float)
MS_REG_GPU_KERNEL_THREE(CombineScaleMomentum,
                        KernelAttr()
                          .AddAllSameAttr(true, 6, true)
                          .AddInputAttr(kNumberTypeFloat32)  // scale
                          .AddInputAttr(kNumberTypeFloat32)  // variable
                          .AddInputAttr(kNumberTypeFloat32)  // accumulation
                          .AddInputAttr(kNumberTypeFloat32)  // learning_rate
                          .AddInputAttr(kNumberTypeFloat16)  // gradient
                          .AddInputAttr(kNumberTypeFloat32)  // momentum
                          .AddOutputAttr(kNumberTypeFloat32),
                        CombineMomentumGpuKernelMod, float, float, half)
MS_REG_GPU_KERNEL_THREE(CombineWeightDecayScaleMomentum,
                        KernelAttr()
                          .AddAllSameAttr(true, 7, true)
                          .AddInputAttr(kNumberTypeFloat32)  // weight decay
                          .AddInputAttr(kNumberTypeFloat32)  // scale
                          .AddInputAttr(kNumberTypeFloat32)  // variable
                          .AddInputAttr(kNumberTypeFloat32)  // accumulation
                          .AddInputAttr(kNumberTypeFloat32)  // learning_rate
                          .AddInputAttr(kNumberTypeFloat32)  // gradient
                          .AddInputAttr(kNumberTypeFloat32)  // momentum
                          .AddOutputAttr(kNumberTypeFloat32),
                        CombineMomentumGpuKernelMod, float, float, float)
MS_REG_GPU_KERNEL_THREE(CombineWeightDecayScaleMomentum,
                        KernelAttr()
                          .AddAllSameAttr(true, 7, true)
                          .AddInputAttr(kNumberTypeFloat32)  // weight decay
                          .AddInputAttr(kNumberTypeFloat32)  // scale
                          .AddInputAttr(kNumberTypeFloat32)  // variable
                          .AddInputAttr(kNumberTypeFloat32)  // accumulation
                          .AddInputAttr(kNumberTypeFloat32)  // learning_rate
                          .AddInputAttr(kNumberTypeFloat16)  // gradient
                          .AddInputAttr(kNumberTypeFloat32)  // momentum
                          .AddOutputAttr(kNumberTypeFloat32),
                        CombineMomentumGpuKernelMod, float, float, half)
}  // namespace kernel
}  // namespace mindspore
