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

#include "backend/kernel_compiler/gpu/math/broadcast_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// fp64
MS_REG_GPU_KERNEL_ONE(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Maximum,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Mod, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  FloorMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Atan2,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, double)
MS_REG_GPU_KERNEL_ONE(
  LessEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, double)

// fp32
MS_REG_GPU_KERNEL_ONE(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Maximum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  FloorDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  DivNoNan,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Mod, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  FloorMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  Atan2,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  LessEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  NotEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  TruncateDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)
MS_REG_GPU_KERNEL_ONE(
  TruncateMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastOpGpuKernelMod, float)

// fp16
MS_REG_GPU_KERNEL_ONE(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Maximum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  FloorDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  DivNoNan,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Mod, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  FloorMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  Atan2,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  LessEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  NotEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  TruncateDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)
MS_REG_GPU_KERNEL_ONE(
  TruncateMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  BroadcastOpGpuKernelMod, half)

// int32
MS_REG_GPU_KERNEL_ONE(
  Greater, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Minimum, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Maximum, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  RealDiv, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  DivNoNan, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  Mod, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  FloorMod, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  LessEqual, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  TruncateDiv,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)
MS_REG_GPU_KERNEL_ONE(
  TruncateMod,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastOpGpuKernelMod, int)

// int64
MS_REG_GPU_KERNEL_ONE(
  Greater, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Less, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Add, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Minimum, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Maximum, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  AbsGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Div, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  DivNoNan, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  Mod, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  FloorMod, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  LessEqual, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int64_t)
MS_REG_GPU_KERNEL_ONE(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int64_t)

// int8
MS_REG_GPU_KERNEL_ONE(
  DivNoNan, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  BroadcastOpGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(
  GreaterEqual, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(
  LessEqual, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  BroadcastOpGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(
  TruncateDiv, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  BroadcastOpGpuKernelMod, int8_t)
MS_REG_GPU_KERNEL_ONE(
  TruncateMod, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  BroadcastOpGpuKernelMod, int8_t)

// uint32
MS_REG_GPU_KERNEL_ONE(
  Sub, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  BroadcastOpGpuKernelMod, uint)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  BroadcastOpGpuKernelMod, uint)

// uint8
MS_REG_GPU_KERNEL_ONE(
  DivNoNan, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  BroadcastOpGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(
  LessEqual, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  BroadcastOpGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(
  TruncateDiv,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  BroadcastOpGpuKernelMod, uint8_t)
MS_REG_GPU_KERNEL_ONE(
  TruncateMod,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  BroadcastOpGpuKernelMod, uint8_t)

// int16
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(
  LessEqual, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, int16_t)
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
  BroadcastOpGpuKernelMod, int16_t)

// uint16
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
  BroadcastOpGpuKernelMod, uint16_t)

// uint32
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  BroadcastOpGpuKernelMod, uint32_t)

// uint64
MS_REG_GPU_KERNEL_ONE(
  Mul, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
  BroadcastOpGpuKernelMod, uint64_t)

// bool
MS_REG_GPU_KERNEL_ONE(
  Equal, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(
  LogicalAnd, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, bool)
MS_REG_GPU_KERNEL_ONE(
  LogicalOr, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  BroadcastOpGpuKernelMod, bool)
}  // namespace kernel
}  // namespace mindspore
