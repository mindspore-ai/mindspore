/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/other/concat_offset_gpu_kernel.h"

namespace mindspore {
namespace kernel {
#define REG_CONCAT_OFFSET_GPU_INPUT_OUTPUT(T0_MS_DTYPE, T0_DTYPE, T1_MS_DTYPE, T1_DTYPE)                        \
  MS_REG_GPU_KERNEL_TWO(ConcatOffset,                                                                           \
                        KernelAttr().AddAllSameAttr(true).AddInputAttr(T0_MS_DTYPE).AddOutputAttr(T1_MS_DTYPE), \
                        ConcatOffsetGpuKernelMod, T0_DTYPE, T1_DTYPE)

#define REG_CONCAT_OFFSET_GPU_INPUT(MS_DTYPE, DTYPE)                             \
  REG_CONCAT_OFFSET_GPU_INPUT_OUTPUT(MS_DTYPE, DTYPE, kNumberTypeInt64, int64_t) \
  REG_CONCAT_OFFSET_GPU_INPUT_OUTPUT(MS_DTYPE, DTYPE, kNumberTypeInt32, int32_t)

#define REG_CONCAT_OFFSET_GPU_INPUT_FLOAT(F) \
  F(kNumberTypeFloat64, double) F(kNumberTypeFloat32, float) F(kNumberTypeFloat16, half)

#define REG_CONCAT_OFFSET_GPU_INPUT_INT(F) \
  F(kNumberTypeInt64, int64_t)             \
  F(kNumberTypeInt32, int32_t)             \
  F(kNumberTypeInt16, int16_t)             \
  F(kNumberTypeInt8, char)                 \
  F(kNumberTypeUInt64, uint64_t)           \
  F(kNumberTypeUInt32, uint32_t)           \
  F(kNumberTypeUInt16, uint16_t)           \
  F(kNumberTypeUInt8, uchar)               \
  F(kNumberTypeBool, bool)

REG_CONCAT_OFFSET_GPU_INPUT_FLOAT(REG_CONCAT_OFFSET_GPU_INPUT)
REG_CONCAT_OFFSET_GPU_INPUT_INT(REG_CONCAT_OFFSET_GPU_INPUT)
}  // namespace kernel
}  // namespace mindspore
