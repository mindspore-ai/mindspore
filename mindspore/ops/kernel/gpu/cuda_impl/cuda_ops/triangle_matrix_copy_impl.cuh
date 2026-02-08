/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRIANGLE_MATRIX_COPY_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRIANGLE_MATRIX_COPY_IMPL_CUH_
#include <cublas_v2.h>
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT cudaError_t TriangleMatrixCopy(const T *input, T *output, bool clean, cublasFillMode_t uplo,
                                               const size_t count, const size_t ldb, const size_t m,
                                               cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t MatrixCopy(const T *input, T *output, const size_t count, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRIANGLE_MATRIX_COPY_IMPL_CUH_
