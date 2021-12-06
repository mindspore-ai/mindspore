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
#include "backend/kernel_compiler/gpu/other/dynamic_broadcast_grad_args_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(DynamicBroadcastGradientArgs,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      DynamicBroadcastGradientArgsGpuKernel, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicBroadcastGradientArgs,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      DynamicBroadcastGradientArgsGpuKernel, int32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicBroadcastGradientArgs,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddOutputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      DynamicBroadcastGradientArgsGpuKernel, uint64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicBroadcastGradientArgs,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddOutputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      DynamicBroadcastGradientArgsGpuKernel, uint32_t, int64_t)
}  // namespace kernel
}  // namespace mindspore
