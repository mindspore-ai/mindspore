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

#include "backend/kernel_compiler/gpu/arrays/unsorted_segment_max_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  UnsortedSegmentMaxGpuKernel, float, int)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  UnsortedSegmentMaxGpuKernel, float, int64_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  UnsortedSegmentMaxGpuKernel, half, int)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentMax,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  UnsortedSegmentMaxGpuKernel, half, int64_t)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentMax,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  UnsortedSegmentMaxGpuKernel, int, int)

MS_REG_GPU_KERNEL_TWO(
  UnsortedSegmentMax,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  UnsortedSegmentMaxGpuKernel, int, int64_t)

// Dynamic Mode - registered for int32/int64 3rd input
MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMaxGpuKernel, float, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMaxGpuKernel, float, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMaxGpuKernel, float, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat32),
                      UnsortedSegmentMaxGpuKernel, float, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMaxGpuKernel, half, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMaxGpuKernel, half, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMaxGpuKernel, half, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      UnsortedSegmentMaxGpuKernel, half, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMaxGpuKernel, int, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMaxGpuKernel, int, int64_t)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMaxGpuKernel, int, int)

MS_REG_GPU_KERNEL_TWO(UnsortedSegmentMax,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      UnsortedSegmentMaxGpuKernel, int, int64_t)
}  // namespace kernel
}  // namespace mindspore
