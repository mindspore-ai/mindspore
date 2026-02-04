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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SPARSE_GATHER_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SPARSE_GATHER_V2_GPU_KERNEL_H_

#include <vector>
#include <utility>
#include "kernel/gpu/cuda/arrays/gather_gpu_kernel.h"

namespace mindspore {
namespace kernel {
class SparseGatherV2GpuKernelMod : public GatherGpuKernelMod {
 public:
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SPARSE_GATHER_V2_GPU_KERNEL_H_
