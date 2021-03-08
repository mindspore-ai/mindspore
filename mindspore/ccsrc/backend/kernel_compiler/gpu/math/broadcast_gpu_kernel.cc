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

#include "backend/kernel_compiler/gpu/math/broadcast_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// fp64
MS_REG_GPU_KERNEL_ONE(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernel, double)
MS_REG_GPU_KERNEL_ONE(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernel, double)

// fp32
MS_REG_GPU_KERNEL_ONE(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Maximum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  FloorDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(
  DivNoNan,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernel, float)

// fp16
MS_REG_GPU_KERNEL_ONE(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Maximum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  FloorDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)
MS_REG_GPU_KERNEL_ONE(
  DivNoNan,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernel, half)

// int32
MS_REG_GPU_KERNEL_ONE(
  Greater, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  Minimum, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  Maximum, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)
MS_REG_GPU_KERNEL_ONE(
  DivNoNan, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernel, int)

// int64
MS_REG_GPU_KERNEL_ONE(
  Greater, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Minimum, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Maximum, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(
  DivNoNan, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernel, int64_t)

// int8
MS_REG_GPU_KERNEL_ONE(
  DivNoNan, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  BroadcastOpGpuKernel, int8_t)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, int8_t)

// uint8
MS_REG_GPU_KERNEL_ONE(
  DivNoNan, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  BroadcastOpGpuKernel, uint8_t)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernel, uint8_t)
}  // namespace kernel
}  // namespace mindspore
