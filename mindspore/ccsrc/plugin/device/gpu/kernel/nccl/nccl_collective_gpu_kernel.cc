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

#include "plugin/device/gpu/kernel/nccl/nccl_collective_gpu_kernel.h"

namespace mindspore {
namespace kernel {
#define MS_REG_GPU_COLLECTIVE_OPS(OPNAME, MS_TYPE, T)                                                           \
  MS_REG_GPU_KERNEL_ONE(OPNAME, KernelAttr().AddAllSameAttr(true).AddInputAttr(MS_TYPE).AddOutputAttr(MS_TYPE), \
                        NcclCollectiveGpuKernel, T)

#define MS_REG_GPU_COLLECTIVE_OPS_DATA_GENERALIZE(OPNAME)        \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeBool, bool)       \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeInt8, int8_t)     \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeInt32, int32_t)   \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeInt64, int64_t)   \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeUInt8, uint8_t)   \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeUInt32, uint32_t) \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeUInt64, uint64_t) \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeFloat16, half)    \
  MS_REG_GPU_COLLECTIVE_OPS(OPNAME, kNumberTypeFloat32, float)

MS_REG_GPU_COLLECTIVE_OPS_DATA_GENERALIZE(AllReduce)
MS_REG_GPU_COLLECTIVE_OPS_DATA_GENERALIZE(AllGather)
MS_REG_GPU_COLLECTIVE_OPS_DATA_GENERALIZE(ReduceScatter)
MS_REG_GPU_COLLECTIVE_OPS_DATA_GENERALIZE(Broadcast)
}  // namespace kernel
}  // namespace mindspore
