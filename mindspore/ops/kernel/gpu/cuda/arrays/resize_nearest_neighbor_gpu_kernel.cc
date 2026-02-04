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

#include "kernel/gpu/cuda/arrays/resize_nearest_neighbor_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(ResizeNearestNeighbor,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeFloat32),
                      ResizeNearestNeighborGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ResizeNearestNeighbor,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeFloat16),
                      ResizeNearestNeighborGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ResizeNearestNeighbor,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeInt32),
                      ResizeNearestNeighborGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(ResizeNearestNeighborV2,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeFloat64),
                      ResizeNearestNeighborGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(ResizeNearestNeighborV2,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeFloat32),
                      ResizeNearestNeighborGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(ResizeNearestNeighborV2,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeFloat16),
                      ResizeNearestNeighborGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(ResizeNearestNeighborV2,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeUInt8),
                      ResizeNearestNeighborGpuKernelMod, uint8_t)
}  // namespace kernel
}  // namespace mindspore
