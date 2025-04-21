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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CTC_GREEDY_DECODER_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CTC_GREEDY_DECODER_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalCTCGreedyDecoder(const T *input, const int bound, const size_t outer_size,
                                                const size_t batch_size, int64_t *decoded_values_temp,
                                                T *log_probability, const uint32_t &device_id,
                                                cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t Calmerge(int64_t *decoded_values_temp, const int32_t *sequence_length,
                                     const size_t batch_size, const int bound, const bool merge_ok, T *log_probability,
                                     int64_t *nums_count, const uint32_t &device_id, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t Calindices(const int64_t *decoded_values_temp, const int64_t *nums_count,
                                       const size_t batch_size, int64_t *decoded_indices, int64_t *decoded_values,
                                       int64_t *decoded_shape, const uint32_t &device_id, cudaStream_t cuda_stream,
                                       int64_t *count);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CTC_GREEDY_DECODER_IMPL_CUH_
