/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/arrays/scatter_nd_functor_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// ScatterNdUpdate
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      ScatterNdFunctorKernel, double, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      ScatterNdFunctorKernel, double, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      ScatterNdFunctorKernel, float, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      ScatterNdFunctorKernel, float, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      ScatterNdFunctorKernel, half, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      ScatterNdFunctorKernel, half, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      ScatterNdFunctorKernel, int, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      ScatterNdFunctorKernel, int, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddOutputAttr(kNumberTypeInt16),
                      ScatterNdFunctorKernel, int16_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddOutputAttr(kNumberTypeInt16),
                      ScatterNdFunctorKernel, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      ScatterNdFunctorKernel, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      ScatterNdFunctorKernel, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      ScatterNdFunctorKernel, int8_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      ScatterNdFunctorKernel, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeBool),
                      ScatterNdFunctorKernel, bool, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdUpdate,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeBool),
                      ScatterNdFunctorKernel, bool, int64_t)

// ScatterNdAdd
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      ScatterNdFunctorKernel, double, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      ScatterNdFunctorKernel, double, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      ScatterNdFunctorKernel, float, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      ScatterNdFunctorKernel, float, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      ScatterNdFunctorKernel, half, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      ScatterNdFunctorKernel, half, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      ScatterNdFunctorKernel, int, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      ScatterNdFunctorKernel, int, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddOutputAttr(kNumberTypeInt16),
                      ScatterNdFunctorKernel, int16_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddOutputAttr(kNumberTypeInt16),
                      ScatterNdFunctorKernel, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      ScatterNdFunctorKernel, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      ScatterNdFunctorKernel, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      ScatterNdFunctorKernel, int8_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      ScatterNdFunctorKernel, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeBool),
                      ScatterNdFunctorKernel, bool, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdAdd,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeBool),
                      ScatterNdFunctorKernel, bool, int64_t)

// ScatterNdSub
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      ScatterNdFunctorKernel, double, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      ScatterNdFunctorKernel, double, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      ScatterNdFunctorKernel, float, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      ScatterNdFunctorKernel, float, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      ScatterNdFunctorKernel, half, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddOutputAttr(kNumberTypeFloat16),
                      ScatterNdFunctorKernel, half, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      ScatterNdFunctorKernel, int, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      ScatterNdFunctorKernel, int, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddOutputAttr(kNumberTypeInt16),
                      ScatterNdFunctorKernel, int16_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddOutputAttr(kNumberTypeInt16),
                      ScatterNdFunctorKernel, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      ScatterNdFunctorKernel, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddOutputAttr(kNumberTypeUInt8),
                      ScatterNdFunctorKernel, uint8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      ScatterNdFunctorKernel, int8_t, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddOutputAttr(kNumberTypeInt8),
                      ScatterNdFunctorKernel, int8_t, int64_t)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeBool),
                      ScatterNdFunctorKernel, bool, int)
MS_REG_GPU_KERNEL_TWO(ScatterNdSub,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeBool)
                        .AddOutputAttr(kNumberTypeBool),
                      ScatterNdFunctorKernel, bool, int64_t)
}  // namespace kernel
}  // namespace mindspore
