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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_SLICE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_SLICE_IMPL_CUH_

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename DataType, typename IndexType>
CUDA_LIB_EXPORT void SparseSlice(const IndexType *indices_ptr, const DataType *values_ptr,
                                 const IndexType *x_ptr, IndexType *start_ptr, IndexType *size_ptr,
                                 IndexType *y_indices_ptr, DataType *y_values_ptr, IndexType *out_shape_ptr,
                                 int64_t *sum_count_ptr, size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                 uint32_t device_id, cudaStream_t cuda_stream);

#endif
