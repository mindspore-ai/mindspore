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

#include "backend/kernel_compiler/gpu/arrays/strided_slice_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      StridedSliceGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      StridedSliceGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      StridedSliceGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      StridedSliceGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      StridedSliceGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      StridedSliceGpuKernel, short)  // NOLINT
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      StridedSliceGpuKernel, int8_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      StridedSliceGpuKernel, uint64_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      StridedSliceGpuKernel, uint32_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      StridedSliceGpuKernel, uint16_t)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      StridedSliceGpuKernel, uchar)
MS_REG_GPU_KERNEL_ONE(StridedSlice, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      StridedSliceGpuKernel, bool)
}  // namespace kernel
}  // namespace mindspore
