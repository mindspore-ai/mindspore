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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LU_UNPACK_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LU_UNPACK_GRAD_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalTrilExpendWidth(const int64_t size, T *l_grad_input, const int64_t matrix_L_height,
                                               const int64_t matrix_L_width, T *l_grad_output,
                                               const int64_t lu_data_height, const int64_t lu_data_width,
                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalTrilLower(const int64_t size, T *l_grad_input, const int64_t matrix_L_height,
                                         const int64_t matrix_L_width, T *l_grad_output, const int64_t lu_data_height,
                                         const int64_t lu_data_width, const uint32_t &device_id,
                                         cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalTriuExpendHeight(const int64_t size, T *u_grad_input, const int64_t matrix_U_height,
                                                const int64_t matrix_U_width, T *u_grad_output,
                                                const int64_t lu_data_height, const int64_t lu_data_width,
                                                const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalTriuUpper(const int64_t size, T *u_grad_input, const int64_t matrix_U_height,
                                         const int64_t matrix_U_width, T *u_grad_output, const int64_t lu_data_height,
                                         const int64_t lu_data_width, const uint32_t &device_id,
                                         cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LU_UNPACK_GRAD_IMPL_CUH_
