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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_SHUFFLE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_SHUFFLE_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT void ScalarShuffle(const int64_t size, const int *perm, const T *input, T *output,
                                   const uint32_t device_id, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT void TensorShuffle(const int64_t shuffle_size, const int64_t inner_size, const int *perm,
                                   const T *input, T *output, const uint32_t device_id, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_SHUFFLE_IMPL_CUH_
