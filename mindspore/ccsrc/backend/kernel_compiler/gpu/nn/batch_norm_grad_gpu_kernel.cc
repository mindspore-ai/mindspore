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

#include "backend/kernel_compiler/gpu/nn/batch_norm_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(BatchNormGrad,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)    // dy
                        .AddInputAttr(kNumberTypeFloat32)    // x
                        .AddInputAttr(kNumberTypeFloat32)    // scale
                        .AddInputAttr(kNumberTypeFloat32)    // save_mean
                        .AddInputAttr(kNumberTypeFloat32)    // save_variance
                        .AddInputAttr(kNumberTypeFloat32)    // reserve
                        .AddOutputAttr(kNumberTypeFloat32)   // dx
                        .AddOutputAttr(kNumberTypeFloat32)   // dscale
                        .AddOutputAttr(kNumberTypeFloat32),  // dbias
                      BatchNormGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(BatchNormGrad,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)    // dy
                        .AddInputAttr(kNumberTypeFloat16)    // x
                        .AddInputAttr(kNumberTypeFloat32)    // scale
                        .AddInputAttr(kNumberTypeFloat32)    // save_mean
                        .AddInputAttr(kNumberTypeFloat32)    // save_variance
                        .AddInputAttr(kNumberTypeFloat32)    // reserve
                        .AddOutputAttr(kNumberTypeFloat16)   // dx
                        .AddOutputAttr(kNumberTypeFloat32)   // dscale
                        .AddOutputAttr(kNumberTypeFloat32),  // dbias
                      BatchNormGradGpuKernel, half)

MS_REG_GPU_KERNEL_ONE(BatchNormGradWithActivation,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)    // dy
                        .AddInputAttr(kNumberTypeFloat32)    // x
                        .AddInputAttr(kNumberTypeFloat32)    // scale
                        .AddInputAttr(kNumberTypeFloat32)    // save_mean
                        .AddInputAttr(kNumberTypeFloat32)    // save_variance
                        .AddInputAttr(kNumberTypeFloat32)    // reserve
                        .AddInputAttr(kNumberTypeFloat32)    // b
                        .AddInputAttr(kNumberTypeFloat32)    // y
                        .AddOutputAttr(kNumberTypeFloat32)   // dx
                        .AddOutputAttr(kNumberTypeFloat32)   // dscale
                        .AddOutputAttr(kNumberTypeFloat32),  // dbias
                      BatchNormGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(BatchNormGradWithActivation,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)    // dy
                        .AddInputAttr(kNumberTypeFloat16)    // x
                        .AddInputAttr(kNumberTypeFloat32)    // scale
                        .AddInputAttr(kNumberTypeFloat32)    // save_mean
                        .AddInputAttr(kNumberTypeFloat32)    // save_variance
                        .AddInputAttr(kNumberTypeFloat32)    // reserve
                        .AddInputAttr(kNumberTypeFloat32)    // b
                        .AddInputAttr(kNumberTypeFloat16)    // y
                        .AddOutputAttr(kNumberTypeFloat16)   // dx
                        .AddOutputAttr(kNumberTypeFloat32)   // dscale
                        .AddOutputAttr(kNumberTypeFloat32),  // dbias
                      BatchNormGradGpuKernel, half)

MS_REG_GPU_KERNEL_ONE(BatchNormGradWithAddAndActivation,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)    // dy
                        .AddInputAttr(kNumberTypeFloat32)    // x
                        .AddInputAttr(kNumberTypeFloat32)    // scale
                        .AddInputAttr(kNumberTypeFloat32)    // save_mean
                        .AddInputAttr(kNumberTypeFloat32)    // save_variance
                        .AddInputAttr(kNumberTypeFloat32)    // reserve
                        .AddInputAttr(kNumberTypeFloat32)    // b
                        .AddInputAttr(kNumberTypeFloat32)    // y
                        .AddOutputAttr(kNumberTypeFloat32)   // dx
                        .AddOutputAttr(kNumberTypeFloat32)   // dscale
                        .AddOutputAttr(kNumberTypeFloat32)   // dbias
                        .AddOutputAttr(kNumberTypeFloat32),  // dz
                      BatchNormGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(BatchNormGradWithAddAndActivation,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)    // dy
                        .AddInputAttr(kNumberTypeFloat16)    // x
                        .AddInputAttr(kNumberTypeFloat32)    // scale
                        .AddInputAttr(kNumberTypeFloat32)    // save_mean
                        .AddInputAttr(kNumberTypeFloat32)    // save_variance
                        .AddInputAttr(kNumberTypeFloat32)    // reserve
                        .AddInputAttr(kNumberTypeFloat32)    // b
                        .AddInputAttr(kNumberTypeFloat16)    // y
                        .AddOutputAttr(kNumberTypeFloat16)   // dx
                        .AddOutputAttr(kNumberTypeFloat32)   // dscale
                        .AddOutputAttr(kNumberTypeFloat32)   // dbias
                        .AddOutputAttr(kNumberTypeFloat16),  // dz
                      BatchNormGradGpuKernel, half)
}  // namespace kernel
}  // namespace mindspore
