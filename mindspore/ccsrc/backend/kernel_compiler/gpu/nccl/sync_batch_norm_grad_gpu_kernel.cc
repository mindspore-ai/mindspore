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

#include "backend/kernel_compiler/gpu/nccl/sync_batch_norm_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_THREE(SyncBatchNormGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32),
                        SyncBatchNormGradGpuKernel, float, float, float)
MS_REG_GPU_KERNEL_THREE(SyncBatchNormGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32),
                        SyncBatchNormGradGpuKernel, half, float, float)
MS_REG_GPU_KERNEL_THREE(SyncBatchNormGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat16),
                        SyncBatchNormGradGpuKernel, float, half, float)
MS_REG_GPU_KERNEL_THREE(SyncBatchNormGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat16),
                        SyncBatchNormGradGpuKernel, half, half, float)
MS_REG_GPU_KERNEL_THREE(SyncBatchNormGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32),
                        SyncBatchNormGradGpuKernel, float, float, half)
MS_REG_GPU_KERNEL_THREE(SyncBatchNormGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32),
                        SyncBatchNormGradGpuKernel, half, float, half)
MS_REG_GPU_KERNEL_THREE(SyncBatchNormGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat16),
                        SyncBatchNormGradGpuKernel, float, half, half)
MS_REG_GPU_KERNEL_THREE(SyncBatchNormGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat16)
                          .AddOutputAttr(kNumberTypeFloat16),
                        SyncBatchNormGradGpuKernel, half, half, half)
}  // namespace kernel
}  // namespace mindspore
