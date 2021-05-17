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

#include "backend/kernel_compiler/gpu/arrays/broadcast_to_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      BroadcastToGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      BroadcastToGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      BroadcastToGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      BroadcastToGpuKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      BroadcastToGpuKernel, int32_t)
MS_REG_GPU_KERNEL_ONE(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      BroadcastToGpuKernel, int64_t)
}  // namespace kernel
}  // namespace mindspore
