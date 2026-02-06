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

#include "kernel/gpu/cuda/random/random_categorical_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, half, int, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, half, int, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, half, int, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, float, int, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, float, int, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, float, int, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, double, int, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, double, int, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, double, int, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, half, int64_t, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, half, int64_t, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, half, int64_t, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, float, int64_t, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, float, int64_t, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, float, int64_t, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, double, int64_t, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, double, int64_t, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, double, int64_t, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, half, int, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, half, int, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, half, int, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, float, int, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, float, int, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, float, int, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, double, int, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, double, int, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, double, int, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, half, int64_t, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, half, int64_t, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat16)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, half, int64_t, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, float, int64_t, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, float, int64_t, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, float, int64_t, int64_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt16),
                        RandomCategoricalGpuKernelMod, double, int64_t, int16_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt32),
                        RandomCategoricalGpuKernelMod, double, int64_t, int32_t)
MS_REG_GPU_KERNEL_THREE(RandomCategorical,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeInt64),
                        RandomCategoricalGpuKernelMod, double, int64_t, int64_t)
}  // namespace kernel
}  // namespace mindspore
