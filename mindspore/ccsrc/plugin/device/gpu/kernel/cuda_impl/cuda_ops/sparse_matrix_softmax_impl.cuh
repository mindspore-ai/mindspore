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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_MATRIX_SOFTMAX_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_MATRIX_SOFTMAX_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename DataType, typename IndexType>
CUDA_LIB_EXPORT cudaError_t SparseMatrixSoftmax(int shape_size, int batch_pointers_size, int row_pointers_size,
                                                int col_indices_size, IndexType *x_dense_shape,
                                                IndexType *x_batch_pointers, IndexType *x_row_pointers,
                                                IndexType *x_col_indices, DataType *x_values, IndexType *y_dense_shape,
                                                IndexType *y_batch_pointers, IndexType *y_row_pointers,
                                                IndexType *y_col_indices, DataType *softmax, uint32_t device_id,
                                                cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_MATRIX_SOFTMAX_IMPL_CUH_
