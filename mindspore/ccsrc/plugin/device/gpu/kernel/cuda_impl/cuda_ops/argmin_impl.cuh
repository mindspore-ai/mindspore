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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ARGMIN_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ARGMIN_IMPL_CUH_
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

#ifdef __cplusplus
extern "C" {
#endif
CUDA_LIB_EXPORT cudaError_t CalArgminFp64Int32(const double *input, const int bound, const size_t outer_size,
                                        const size_t inner_size, int *output, const uint32_t &device_id,
                                        cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalArgminFp32Int32(const float *input, const int bound, const size_t outer_size,
                                        const size_t inner_size, int *output, const uint32_t &device_id,
                                        cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalArgminFp16Int32(const half *input, const int bound, const size_t outer_size,
                                        const size_t inner_size, int *output, const uint32_t &device_id,
                                        cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalArgminFp64Int64(const double *input, const int64_t bound, const size_t outer_size,
                                        const size_t inner_size, int64_t *output, const uint32_t &device_id,
                                        cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalArgminFp32Int64(const float *input, const int64_t bound, const size_t outer_size,
                                        const size_t inner_size, int64_t *output, const uint32_t &device_id,
                                        cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalArgminFp16Int64(const half *input, const int64_t bound, const size_t outer_size,
                                        const size_t inner_size, int64_t *output, const uint32_t &device_id,
                                        cudaStream_t cuda_stream);
#ifdef __cplusplus
}
#endif

template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t CalArgmin(const T *input, const S bound, const size_t outer_size, const size_t inner_size,
                               S *output, const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ARGMIN_IMPL_CUH_
