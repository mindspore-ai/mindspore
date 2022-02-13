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

#include "plugin/device/gpu/kernel/arrays/cast_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CastGpuKernelMod,
                      int8_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, int8_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, int8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, int8_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, int8_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, int8_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, int8_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, int8_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, int8_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, int8_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool), CastGpuKernelMod,
                      int8_t, bool)
template <typename T>
using Complex = mindspore::utils::Complex<T>;
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, int8_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, int8_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, int16_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, int16_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, int16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, int16_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, int16_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, int16_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, int16_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, int16_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, int16_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, int16_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, int16_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, int16_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, int16_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, int32_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, int32_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, int32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, int32_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, int32_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, int32_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, int32_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, int32_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, int32_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, int32_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, int32_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, int32_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, int32_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, int64_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, int64_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, int64_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, int64_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, int64_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, int64_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, int64_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, int64_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, int64_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, int64_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, int64_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, int64_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, int64_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, uint8_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, uint8_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, uint8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, uint8_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, uint8_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, uint8_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, uint8_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, uint8_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, uint8_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, uint8_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, uint8_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, uint8_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, uint8_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, uint16_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, uint16_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, uint16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, uint16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, uint16_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, uint16_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, uint16_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, uint16_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, uint16_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, uint16_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, uint16_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, uint16_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, uint16_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, uint16_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, uint32_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, uint32_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, uint32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, uint32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, uint32_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, uint32_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, uint32_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, uint32_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, uint32_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, uint32_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, uint32_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, uint32_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, uint32_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, uint32_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, uint64_t, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, uint64_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, uint64_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, uint64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, uint64_t, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, uint64_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, uint64_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, uint64_t, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, uint64_t, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, uint64_t, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, uint64_t, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, uint64_t, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, uint64_t, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, uint64_t, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, half, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, half, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, half, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, half, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, half, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, half, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, half, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, half, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, half, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, half, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, half, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, half, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, float, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, float, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, float, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, float, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, float, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, float, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, float, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, float, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, float, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, float, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, float, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, float, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, double, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, double, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, double, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, double, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, double, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, double, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, double, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, double, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, double, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, double, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, double, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, double, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt8), CastGpuKernelMod,
                      bool, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, bool, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, bool, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, bool, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, bool, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, bool, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, bool, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, bool, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, bool, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, bool, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, bool, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), CastGpuKernelMod,
                      bool, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, bool, Complex<float>)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, bool, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, Complex<float>, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, Complex<float>, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, Complex<float>, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, Complex<float>, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, Complex<float>, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, Complex<float>, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, Complex<float>, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, Complex<float>, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, Complex<float>, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, Complex<float>, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, Complex<float>, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, Complex<float>, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex128),
                      CastGpuKernelMod, Complex<float>, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt8),
                      CastGpuKernelMod, Complex<double>, int8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt16),
                      CastGpuKernelMod, Complex<double>, int16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt32),
                      CastGpuKernelMod, Complex<double>, int32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt64),
                      CastGpuKernelMod, Complex<double>, int64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeUInt8),
                      CastGpuKernelMod, Complex<double>, uint8_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeUInt16),
                      CastGpuKernelMod, Complex<double>, uint16_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeUInt32),
                      CastGpuKernelMod, Complex<double>, uint32_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeUInt64),
                      CastGpuKernelMod, Complex<double>, uint64_t)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat32),
                      CastGpuKernelMod, Complex<double>, float)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
                      CastGpuKernelMod, Complex<double>, double)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat16),
                      CastGpuKernelMod, Complex<double>, half)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeBool),
                      CastGpuKernelMod, Complex<double>, bool)
MS_REG_GPU_KERNEL_TWO(Cast, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex64),
                      CastGpuKernelMod, Complex<double>, Complex<float>)

}  // namespace kernel
}  // namespace mindspore
