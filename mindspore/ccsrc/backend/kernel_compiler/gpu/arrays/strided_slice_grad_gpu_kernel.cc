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

#include "backend/kernel_compiler/gpu/arrays/strided_slice_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      StridedSliceGradGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      StridedSliceGradGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      StridedSliceGradGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      StridedSliceGradGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      StridedSliceGradGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      StridedSliceGradGpuKernel, short)  // NOLINT
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      StridedSliceGradGpuKernel, int8_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      StridedSliceGradGpuKernel, uint64_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      StridedSliceGradGpuKernel, uint32_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      StridedSliceGradGpuKernel, uint16_t)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      StridedSliceGradGpuKernel, uchar)
MS_REG_GPU_KERNEL_ONE(StridedSliceGrad, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      StridedSliceGradGpuKernel, bool)
}  // namespace kernel
}  // namespace mindspore
