/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/slice_gpu_kernel.h"

namespace mindspore {
namespace kernel {
#define REG_SLICE_GPU(MS_DTYPE, DTYPE) \
  MS_REG_GPU_KERNEL_ONE(Slice, KernelAttr().AddInputAttr(MS_DTYPE).AddOutputAttr(MS_DTYPE), SliceFwdGpuKernelMod, DTYPE)

#define REG_SLICE_GPU_DTYPES(F) \
  F(kNumberTypeFloat64, double) \
  F(kNumberTypeFloat32, float)  \
  F(kNumberTypeFloat16, half)   \
  F(kNumberTypeInt64, int64_t)  \
  F(kNumberTypeInt32, int32_t)  \
  F(kNumberTypeInt16, int16_t)  \
  F(kNumberTypeUInt8, uchar)    \
  F(kNumberTypeBool, bool)

REG_SLICE_GPU_DTYPES(REG_SLICE_GPU)

#define REG_DYNAMIC_SLICE_GPU_ATTR(T0_MS_DTYPE, T0_DTYPE, T1_MS_DTYPE, T1_DTYPE) \
  MS_REG_GPU_KERNEL_TWO(Slice,                                                   \
                        KernelAttr()                                             \
                          .AddInputAttr(T0_MS_DTYPE)                             \
                          .AddInputAttr(T1_MS_DTYPE)                             \
                          .AddInputAttr(T1_MS_DTYPE)                             \
                          .AddOutputAttr(T0_MS_DTYPE),                           \
                        SliceFwdGpuKernelMod, T0_DTYPE, T1_DTYPE)

#define REG_DYNAMIC_SLICE_GPU(MS_DTYPE, DTYPE)                           \
  REG_DYNAMIC_SLICE_GPU_ATTR(MS_DTYPE, DTYPE, kNumberTypeInt32, int32_t) \
  REG_DYNAMIC_SLICE_GPU_ATTR(MS_DTYPE, DTYPE, kNumberTypeInt64, int64_t)

REG_SLICE_GPU_DTYPES(REG_DYNAMIC_SLICE_GPU)
}  // namespace kernel
}  // namespace mindspore
