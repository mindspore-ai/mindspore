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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_FILL_EMPTY_ROWS_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_FILL_EMPTY_ROWS_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename S>
CUDA_LIB_EXPORT cudaError_t SparseFillEmptyRows(int64_t *indices_ptr, S *values_ptr, S *default_value,
                                                int64_t *dense_shape_ptr, int device_id, int indice_num,
                                                size_t dense_row, int64_t *elements_per_rows, int *empty_row_count_sum,
                                                int64_t *row_indices, int64_t *input_row_ends, int64_t *sorted_indices,
                                                size_t *final_shape, int64_t *origin_index, int64_t *sorted_key,
                                                cudaStream_t cuda_stream, int64_t *output_indices_ptr,
                                                S *output_values_ptr, bool *output_empty_row_indicator_ptr,
                                                int64_t *output_reverse_index_map_ptr);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_FILL_EMPTY_ROWS_CUH_
