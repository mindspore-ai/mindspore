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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CSR_SPARSE_MATRIX_TO_SPARSE_TENSOR_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CSR_SPARSE_MATRIX_TO_SPARSE_TENSOR_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename S>
CUDA_LIB_EXPORT void CallStackIndices2D(const S *row_indices, const S *col_indices, S *indices, int size,
                                        cudaStream_t cuda_stream);

template <typename S>
CUDA_LIB_EXPORT void CallStackIndices3D(const S *batch_pointers, const S *row_indices, const S *col_indices, S *indices,
                                        int batch_size, int total_nnz, size_t shared_memory_size,
                                        cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CSR_SPARSE_MATRIX_TO_SPARSE_TENSOR_CUH_
