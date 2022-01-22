/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/gpu/other/dynamic_reshape_gpu_kernel.h"
#include <iterator>
#include <algorithm>
#include <functional>
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  DynamicReshape,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  DynamicReshapeKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(
  DynamicReshape,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  DynamicReshapeKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(
  DynamicReshape,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  DynamicReshapeKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(
  DynamicReshape,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  DynamicReshapeKernelMod, int64_t, int)
MS_REG_GPU_KERNEL_TWO(
  DynamicReshape,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  DynamicReshapeKernelMod, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicReshape,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  DynamicReshapeKernelMod, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicReshape,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  DynamicReshapeKernelMod, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicReshape,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  DynamicReshapeKernelMod, int, int64_t)
}  // namespace kernel
}  // namespace mindspore
