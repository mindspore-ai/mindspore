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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_IGAMMA_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_IGAMMA_IMPL_CUH_
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

constexpr int64_t kLgammaSameShape = 0;
constexpr int64_t kLgammaAOneElement = 1;
constexpr int64_t kLgammaXOneElement = 2;

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalIgamma(const size_t size, const int64_t type, const T *a, const T *x, T *output,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalIgammac(const size_t size, const int64_t type, const T *a, const T *x, T *output,
                                       const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalIgammaGradA(const size_t size, const int64_t type, const T *a, const T *x, T *output,
                                           const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalBroadcastIgamma(const std::vector<size_t> &inputa_shape,
                                               const std::vector<size_t> &inputx_shape,
                                               const std::vector<size_t> &output_shape, const T *inputa,
                                               const T *inputx, T *output, const uint32_t &device_id,
                                               cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalBroadcastIgammac(const std::vector<size_t> &inputa_shape,
                                                const std::vector<size_t> &inputx_shape,
                                                const std::vector<size_t> &output_shape, const T *inputa,
                                                const T *inputx, T *output, const uint32_t &device_id,
                                                cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalBroadcastIgammaGradA(const std::vector<size_t> &inputa_shape,
                                                    const std::vector<size_t> &inputx_shape,
                                                    const std::vector<size_t> &output_shape, const T *inputa,
                                                    const T *inputx, T *output, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_IGAMMA_IMPL_CUH_
