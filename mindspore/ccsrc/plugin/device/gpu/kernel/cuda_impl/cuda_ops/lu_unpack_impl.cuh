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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LUUNPACK_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LUUNPACK_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t TransposeEyeMatrix(S *lu_pivots_ptr, T *pivots_ptr, S *final_order, const int64_t batch_num,
                                               const int64_t lu_data_dim1, const int64_t lu_pivots_dim,
                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalTrilAux(const size_t size, const T *input, const int64_t out_row, const int64_t out_col,
                                       const int64_t lu_row, const int64_t lu_col, T *output, const uint32_t &device_id,
                                       cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalTriuAux(const size_t size, const T *input, const int64_t out_row, const int64_t out_col,
                                       const int64_t lu_row, const int64_t lu_col, T *output, const uint32_t &device_id,
                                       cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LUUNPACK_IMPL_CUH_
