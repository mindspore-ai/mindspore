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

#include "backend/kernel_compiler/gpu/arrays/strided_slice_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      StridedSliceGradGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      StridedSliceGradGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      StridedSliceGradGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      StridedSliceGradGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      StridedSliceGradGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      StridedSliceGradGpuKernelMod, short)  // NOLINT
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      StridedSliceGradGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      StridedSliceGradGpuKernelMod, uint64_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      StridedSliceGradGpuKernelMod, uint32_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      StridedSliceGradGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      StridedSliceGradGpuKernelMod, uchar)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      StridedSliceGradGpuKernelMod, bool)
}  // namespace kernel
}  // namespace mindspore
