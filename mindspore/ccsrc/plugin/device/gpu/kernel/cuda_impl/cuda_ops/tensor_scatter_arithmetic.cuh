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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TENSOR_SCATTER_ARITHMETIC_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TENSOR_SCATTER_ARITHMETIC_CUH_

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
enum TensorScatterArithmeticFunctionType {
  TENSOR_SCATTER_FUNC_UPDATE = 0,
  TENSOR_SCATTER_FUNC_MIN,
  TENSOR_SCATTER_FUNC_MAX,
  TENSOR_SCATTER_FUNC_ADD,
  TENSOR_SCATTER_FUNC_SUB,
  TENSOR_SCATTER_FUNC_MUL,
  TENSOR_SCATTER_FUNC_DIV,
  TENSOR_SCATTER_FUNC_INVALID_TYPE = 255
};

template <typename T, typename S>
CUDA_LIB_EXPORT void TensorScatterArithmetic(const enum TensorScatterArithmeticFunctionType &func_type, const T *input,
                                             const S *indices, const T *update, T *output, int *has_error,
                                             const size_t &block_size, const size_t &input_size,
                                             const size_t &output_size, const size_t &indices_dim_0,
                                             const size_t &indices_dim_1, S *indices_stride, S *work_shape,
                                             uint32_t device_id, cudaStream_t stream);
template <typename T, typename S>
CUDA_LIB_EXPORT void CallTensorScatterUpdate(const T *input, const S *indices, const T *update, T *output,
                                             int *has_error, const size_t &block_size, const size_t &input_size,
                                             const size_t &output_size, const size_t &indices_dim_0,
                                             const size_t &indices_dim_1, S *indices_stride, S *work_shape,
                                             uint32_t device_id, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TENSOR_SCATTER_ARITHMETIC_CUH_
