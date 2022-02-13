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

#include "plugin/device/gpu/kernel/nn/flatten_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// float64
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      FlattenFwdGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      FlattenFwdGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      FlattenFwdGpuKernelMod, double)
// float32
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      FlattenFwdGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      FlattenFwdGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      FlattenFwdGpuKernelMod, float)
// float16
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      FlattenFwdGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      FlattenFwdGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      FlattenFwdGpuKernelMod, half)
// int64
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      FlattenFwdGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      FlattenFwdGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      FlattenFwdGpuKernelMod, int64_t)
// int32
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      FlattenFwdGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      FlattenFwdGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      FlattenFwdGpuKernelMod, int)
// int16
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      FlattenFwdGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      FlattenFwdGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      FlattenFwdGpuKernelMod, int16_t)
// int8
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      FlattenFwdGpuKernelMod, char)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      FlattenFwdGpuKernelMod, char)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      FlattenFwdGpuKernelMod, char)

// uint64
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      FlattenFwdGpuKernelMod, uint64_t)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      FlattenFwdGpuKernelMod, uint64_t)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      FlattenFwdGpuKernelMod, uint64_t)

// uint32
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      FlattenFwdGpuKernelMod, uint)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      FlattenFwdGpuKernelMod, uint)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      FlattenFwdGpuKernelMod, uint)

// uint16
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      FlattenFwdGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      FlattenFwdGpuKernelMod, uint16_t)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      FlattenFwdGpuKernelMod, uint16_t)

// uint8
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      FlattenFwdGpuKernelMod, uchar)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      FlattenFwdGpuKernelMod, uchar)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      FlattenFwdGpuKernelMod, uchar)

// bool
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      FlattenFwdGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      FlattenFwdGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      FlattenFwdGpuKernelMod, bool)
}  // namespace kernel
}  // namespace mindspore
