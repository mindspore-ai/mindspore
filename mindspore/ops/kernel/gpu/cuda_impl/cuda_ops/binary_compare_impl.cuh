/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_COMPARE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_COMPARE_IMPL_CUH_
#include <vector>
#include "kernel/gpu/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_common.cuh"

#define REGISTER_COMPARE_MIX_TYPE(Op, In0_t, In1_t)                                                                \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<Op, In0_t, In1_t, bool>(                      \
    const bool, const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &, In0_t *, \
    In1_t *, bool *, size_t, cudaStream_t);                                                                        \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<Op, In1_t, In0_t, bool>(                      \
    const bool, const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &, In1_t *, \
    In0_t *, bool *, size_t, cudaStream_t);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_COMPARE_IMPL_CUH_
