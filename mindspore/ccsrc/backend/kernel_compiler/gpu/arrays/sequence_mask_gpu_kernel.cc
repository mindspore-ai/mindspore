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

#include "backend/kernel_compiler/gpu/arrays/sequence_mask_gpu_kernel.h"

namespace mindspore {
namespace kernel {

// keep this as TWO but output is always bool, just in case framework can
// support passing optional dtype and then we can be identical to tf
MS_REG_GPU_KERNEL_TWO(
  SequenceMask,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  SequenceMaskGpuKernel, int32_t, bool)

MS_REG_GPU_KERNEL_TWO(
  SequenceMask,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  SequenceMaskGpuKernel, int64_t, bool)
}  // namespace kernel
}  // namespace mindspore
