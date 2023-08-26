/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_OP_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_OP_IMPL_CUH_
#include <curand_kernel.h>
#include <random>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t StandardNormal(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState,
                                           T *output, size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t UniformInt(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState,
                                       T *input1, size_t input_size_1, T *input2, size_t input_size_2, T *output,
                                       size_t count, cudaStream_t cuda_stream, bool *host_error_res);
template <typename T>
CUDA_LIB_EXPORT cudaError_t UniformReal(uint64_t seed, uint64_t seed_offset, curandStatePhilox4_32_10_t *globalState,
                                        T *output, size_t count, cudaStream_t cuda_stream);
template <typename S>
CUDA_LIB_EXPORT cudaError_t TruncatedNormal(uint64_t seed, uint64_t seed_offset, curandState *globalState, S *output,
                                            size_t count, cudaStream_t cuda_stream);
template <typename R, typename T>
CUDA_LIB_EXPORT cudaError_t RandomPoisson(uint64_t seed, uint64_t seed_offset, curandState *globalState, R *rate,
                                          int64_t rate_size, T *output, size_t count, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t StandardLaplace(uint64_t seed, uint64_t seed_offset, curandState *globalState, T *output,
                                            size_t count, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RANDOM_OP_IMPL_CUH_
