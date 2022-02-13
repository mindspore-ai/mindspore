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

#include "plugin/device/gpu/kernel/arrays/embedding_lookup_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  EmbeddingLookupKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  EmbeddingLookupKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  EmbeddingLookupKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  EmbeddingLookupKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  EmbeddingLookupKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  EmbeddingLookupKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  EmbeddingLookupKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  EmbeddingLookupKernelMod, int, int64_t)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  EmbeddingLookupKernelMod, int16_t, int)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  EmbeddingLookupKernelMod, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
  EmbeddingLookupKernelMod, int8_t, int)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
  EmbeddingLookupKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
  EmbeddingLookupKernelMod, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
  EmbeddingLookupKernelMod, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  EmbeddingLookupKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  EmbeddingLookupKernelMod, bool, int64_t)
// dynamic shape
MS_REG_GPU_KERNEL_TWO(EmbeddingLookup,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      EmbeddingLookupKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(EmbeddingLookup,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      EmbeddingLookupKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(EmbeddingLookup,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      EmbeddingLookupKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(EmbeddingLookup,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      EmbeddingLookupKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(EmbeddingLookup,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      EmbeddingLookupKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(EmbeddingLookup,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      EmbeddingLookupKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(EmbeddingLookup,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeBool),
                      EmbeddingLookupKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(EmbeddingLookup,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeBool),
                      EmbeddingLookupKernelMod, bool, int64_t)
// dynamic shape ends
}  // namespace kernel
}  // namespace mindspore
