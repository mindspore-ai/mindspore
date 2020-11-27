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
#include <cstdint>

#include "backend/kernel_compiler/gpu/arrays/dynamic_shape_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      DynamicShapeGpuKernel, int32_t, int32_t)

MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
                      DynamicShapeGpuKernel, half, int32_t)

MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
                      DynamicShapeGpuKernel, float, int32_t)

MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
                      DynamicShapeGpuKernel, bool, int32_t)

MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
                      DynamicShapeGpuKernel, int32_t, int64_t)

MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
                      DynamicShapeGpuKernel, half, int64_t)

MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
                      DynamicShapeGpuKernel, float, int64_t)

MS_REG_GPU_KERNEL_TWO(DynamicShape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
                      DynamicShapeGpuKernel, bool, int64_t)
}  // namespace kernel
}  // namespace mindspore
