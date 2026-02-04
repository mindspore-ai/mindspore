/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/sparse_gather_v2_gpu_kernel.h"
#include <memory>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
#define REG_INDEX(INPUT_DT, INDEX_DT, INPUT_T, INDEX_T)           \
  {                                                               \
    KernelAttr()                                                  \
      .AddInputAttr(INPUT_DT)                                     \
      .AddInputAttr(INDEX_DT)                                     \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)          \
      .AddOutputAttr(INPUT_DT),                                   \
      &SparseGatherV2GpuKernelMod::LaunchKernel<INPUT_T, INDEX_T> \
  }

#define GATHER_GPU_REGISTER(DT, T) \
  REG_INDEX(DT, kNumberTypeInt64, T, int64_t), REG_INDEX(DT, kNumberTypeInt32, T, int32_t)

template <typename T>
using Complex = mindspore::utils::Complex<T>;

const std::vector<std::pair<KernelAttr, SparseGatherV2GpuKernelMod::KernelRunFunc>>
  &SparseGatherV2GpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SparseGatherV2GpuKernelMod::KernelRunFunc>> func_list = {
    GATHER_GPU_REGISTER(kNumberTypeComplex64, Complex<float>),
    GATHER_GPU_REGISTER(kNumberTypeComplex128, Complex<double>),
    GATHER_GPU_REGISTER(kNumberTypeFloat16, half),
    GATHER_GPU_REGISTER(kNumberTypeFloat32, float),
    GATHER_GPU_REGISTER(kNumberTypeFloat64, double),
    GATHER_GPU_REGISTER(kNumberTypeInt8, uchar),
    GATHER_GPU_REGISTER(kNumberTypeInt16, int16_t),
    GATHER_GPU_REGISTER(kNumberTypeInt32, int32_t),
    GATHER_GPU_REGISTER(kNumberTypeInt64, int64_t),
    GATHER_GPU_REGISTER(kNumberTypeUInt8, uint8_t),
    GATHER_GPU_REGISTER(kNumberTypeUInt16, uint16_t),
    GATHER_GPU_REGISTER(kNumberTypeUInt32, uint32_t),
    GATHER_GPU_REGISTER(kNumberTypeUInt64, uint64_t),
    GATHER_GPU_REGISTER(kNumberTypeBool, bool)};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseGatherV2, SparseGatherV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
