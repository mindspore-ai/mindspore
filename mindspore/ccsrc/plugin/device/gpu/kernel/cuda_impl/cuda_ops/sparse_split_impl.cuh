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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_Split_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_Split_IMPL_CUH_

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename DataType, typename IndexType>
CUDA_LIB_EXPORT void SparseSplit(IndexType *split_dim_ptr, IndexType *indices_ptr, DataType *values_ptr,
                                 IndexType *shape_ptr, IndexType num_split, IndexType **y_indices_ptr,
                                 DataType **y_values_ptr, IndexType *out_shape_ptr, int *sum_count_ptr,
                                 size_t input_nnz_, size_t num_dim_, IndexType *d_block_ptr, cudaStream_t cuda_stream);

#endif
