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

#include "plugin/device/gpu/kernel/arrays/topk_gpu_kernel.h"
#include <algorithm>
#include <map>

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(TopK,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeInt32),
                      TopKGpuKernelMod, float, int)

MS_REG_GPU_KERNEL_TWO(TopK,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeInt32),
                      TopKGpuKernelMod, half, int)

MS_REG_GPU_KERNEL_TWO(TopK,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      TopKGpuKernelMod, int, int)

MS_REG_GPU_KERNEL_TWO(TopK,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeInt32),
                      TopKGpuKernelMod, float, int)

MS_REG_GPU_KERNEL_TWO(TopK,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeInt32),
                      TopKGpuKernelMod, half, int)

MS_REG_GPU_KERNEL_TWO(TopK,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      TopKGpuKernelMod, int, int)
}  // namespace kernel
}  // namespace mindspore
