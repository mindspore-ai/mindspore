/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <cstdint>

#include "plugin/device/gpu/kernel/arrays/dynamic_shape_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, uint8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, uint16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, uint32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, uint64_t, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, int8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, int16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, int64_t, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, bool, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, uint16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, uint32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, uint64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, int32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, bool, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, Complex<float>, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, Complex<float>, int64_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, Complex<double>, int32_t)
MS_REG_GPU_KERNEL_TWO(TensorShape, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, Complex<double>, int64_t)

MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, uint8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, uint16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, uint32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, uint64_t, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, int8_t, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, int16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, int64_t, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, half, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, float, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, double, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, bool, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, uint16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, uint32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, uint64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, int32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, half, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, bool, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, Complex<float>, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, Complex<float>, int64_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt32),
                      TensorShapeGpuKernelMod, Complex<double>, int32_t)
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt64),
                      TensorShapeGpuKernelMod, Complex<double>, int64_t)
}  // namespace kernel
}  // namespace mindspore
