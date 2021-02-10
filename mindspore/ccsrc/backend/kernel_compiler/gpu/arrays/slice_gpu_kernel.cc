/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/arrays/slice_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      SliceGpuFwdKernel, double)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      SliceGpuFwdKernel, float)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      SliceGpuFwdKernel, half)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      SliceGpuFwdKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      SliceGpuFwdKernel, int)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      SliceGpuFwdKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      SliceGpuFwdKernel, uchar)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      SliceGpuFwdKernel, bool)
}  // namespace kernel
}  // namespace mindspore
