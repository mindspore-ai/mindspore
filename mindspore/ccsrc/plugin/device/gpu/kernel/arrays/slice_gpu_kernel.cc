/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/slice_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      SliceFwdGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      SliceFwdGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      SliceFwdGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      SliceFwdGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      SliceFwdGpuKernelMod, int32_t)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      SliceFwdGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      SliceFwdGpuKernelMod, uchar)
MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      SliceFwdGpuKernelMod, bool)
}  // namespace kernel
}  // namespace mindspore
