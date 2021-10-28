/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/arrays/cast_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      int8_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      int8_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      int8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      int8_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt16), CastGpuKernel,
                      int8_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt32), CastGpuKernel,
                      int8_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt64), CastGpuKernel,
                      int8_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32), CastGpuKernel,
                      int8_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat64), CastGpuKernel,
                      int8_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat16), CastGpuKernel,
                      int8_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      int8_t, bool)
template <typename T>
using Complex = mindspore::utils::Complex<T>;
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, int8_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, int8_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      int16_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      int16_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      int16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      int16_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt16), CastGpuKernel,
                      int16_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt32), CastGpuKernel,
                      int16_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt64), CastGpuKernel,
                      int16_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, int16_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, int16_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, int16_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      int16_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, int16_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, int16_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      int32_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      int32_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      int32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      int32_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16), CastGpuKernel,
                      int32_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32), CastGpuKernel,
                      int32_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64), CastGpuKernel,
                      int32_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, int32_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, int32_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, int32_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      int32_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, int32_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, int32_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      int64_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      int64_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      int64_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      int64_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16), CastGpuKernel,
                      int64_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32), CastGpuKernel,
                      int64_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64), CastGpuKernel,
                      int64_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, int64_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, int64_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, int64_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      int64_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, int64_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, int64_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      uint8_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      uint8_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      uint8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      uint8_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt16), CastGpuKernel,
                      uint8_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt32), CastGpuKernel,
                      uint8_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt64), CastGpuKernel,
                      uint8_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, uint8_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, uint8_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, uint8_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      uint8_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, uint8_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, uint8_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      uint16_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      uint16_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      uint16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      uint16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      uint16_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernel, uint16_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernel, uint16_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernel, uint16_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, uint16_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, uint16_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, uint16_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      uint16_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, uint16_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, uint16_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      uint32_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      uint32_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      uint32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      uint32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      uint32_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernel, uint32_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernel, uint32_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernel, uint32_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, uint32_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, uint32_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, uint32_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      uint32_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, uint32_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, uint32_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      uint64_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      uint64_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      uint64_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      uint64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      uint64_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernel, uint64_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernel, uint64_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernel, uint64_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, uint64_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, uint64_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, uint64_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      uint64_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, uint64_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, uint64_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      half, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernel, half, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernel, half, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernel, half, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernel, half, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernel, half, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernel, half, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernel, half, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, half, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, half, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      half, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, half, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, half, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      float, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernel, float, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernel, float, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernel, float, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernel, float, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernel, float, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernel, float, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernel, float, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, float, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, float, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      float, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, float, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, float, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      double, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernel, double, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernel, double, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernel, double, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernel, double, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernel, double, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernel, double, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernel, double, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, double, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, double, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, double, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      double, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, double, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, double, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt8), CastGpuKernel,
                      bool, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt16), CastGpuKernel,
                      bool, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32), CastGpuKernel,
                      bool, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64), CastGpuKernel,
                      bool, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt8), CastGpuKernel,
                      bool, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt16), CastGpuKernel,
                      bool, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt32), CastGpuKernel,
                      bool, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt64), CastGpuKernel,
                      bool, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32), CastGpuKernel,
                      bool, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat64), CastGpuKernel,
                      bool, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat16), CastGpuKernel,
                      bool, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), CastGpuKernel,
                      bool, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, bool, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, bool, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernel, Complex<float>, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernel, Complex<float>, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernel, Complex<float>, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernel, Complex<float>, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernel, Complex<float>, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernel, Complex<float>, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernel, Complex<float>, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernel, Complex<float>, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, Complex<float>, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, Complex<float>, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, Complex<float>, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernel, Complex<float>, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernel, Complex<float>, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernel, Complex<double>, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernel, Complex<double>, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernel, Complex<double>, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernel, Complex<double>, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernel, Complex<double>, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernel, Complex<double>, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernel, Complex<double>, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernel, Complex<double>, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernel, Complex<double>, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernel, Complex<double>, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernel, Complex<double>, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernel, Complex<double>, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernel, Complex<double>, Complex<float>)
}  // namespace kernel
}  // namespace mindspore
