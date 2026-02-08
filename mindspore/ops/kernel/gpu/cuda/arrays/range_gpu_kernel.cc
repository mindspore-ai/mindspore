/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <cstdint>

#include "kernel/gpu/cuda/arrays/range_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Range,
                      KernelAttr()
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      RangeGpuKernelMod, float)

MS_REG_GPU_KERNEL_ONE(Range,
                      KernelAttr()
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      RangeGpuKernelMod, double)

MS_REG_GPU_KERNEL_ONE(Range,
                      KernelAttr()
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      RangeGpuKernelMod, int32_t)

MS_REG_GPU_KERNEL_ONE(Range,
                      KernelAttr()
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      RangeGpuKernelMod, int64_t)
}  // namespace kernel
}  // namespace mindspore
