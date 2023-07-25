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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_DENSE_CWISE_OPERATION_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_DENSE_CWISE_OPERATION_IMPL_CUH_
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

enum SparseDenseCwiseOperationFunctionType {
  SPARSE_DENSE_CWISE_OPERATION_FUNC_ADD = 0,
  SPARSE_DENSE_CWISE_OPERATION_FUNC_MUL,
  SPARSE_DENSE_CWISE_OPERATION_FUNC_DIV,
  SPARSE_DENSE_CWISE_OPERATION_INVALID_TYPE = 255
};

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const T *x1_values,
  const int64_t *x1_shape, const T *x2, T *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const T *x1_values,
  const int64_t *x1_shape, const T *x2, T *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_DENSE_CWISE_OPERATION_IMPL_CUH_
