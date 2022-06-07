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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MATRIX_BAND_PART_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MATRIX_BAND_PART_IMPL_CUH_
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT void MatrixBandPart(const size_t size, const T *x_ptr, const size_t m, const size_t n,
                                    const int64_t lower, const int64_t upper, T *output_ptr, const uint32_t &device_id,
                                    cudaStream_t cuda_stream);

template <typename T, typename LU>
void MatrixBandPartBroadcast(const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
                             const std::vector<size_t> &broadcast_lower_shape,
                             const std::vector<size_t> &broadcast_upper_shape,
                             const std::vector<size_t> &broadcast_output_shape, const T *x_ptr, const size_t m,
                             const size_t n, const LU *lower_ptr, const LU *upper_ptr, T *output_ptr,
                             const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MATRIX_BAND_PART_IMPL_CUH_
