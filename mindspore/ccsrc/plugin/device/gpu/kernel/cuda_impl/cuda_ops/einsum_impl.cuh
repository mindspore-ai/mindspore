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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EINSUM_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EINSUM_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "include/cuda_fp16.h"
#define EINSUM_MAX_DIMENSION 20
template <typename T>
struct DynamicSharedMem;
template <>
struct DynamicSharedMem<double> {
  __device__ double *addr() {
    extern __shared__ double addr_double[];
    return addr_double;
  }
};
template <>
struct DynamicSharedMem<float> {
  __device__ float *addr() {
    extern __shared__ float addr_float[];
    return addr_float;
  }
};
template <>
struct DynamicSharedMem<half> {
  __device__ half *addr() {
    extern __shared__ half addr_half[];
    return addr_half;
  }
};
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalDiagonal(const size_t size, const T *input, const size_t *input_shape,
                                        const size_t shape_size, const size_t left_dim, const size_t right_dim,
                                        T *output, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalDiagonalGrad(const size_t d_size, const T *dout, const size_t *input_shape,
                                            const size_t shape_size, const size_t left_dim, const size_t right_dim,
                                            T *d_inp, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalDot(const size_t size, T *input_a, const T *input_b, T *output,
                                   cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalDotGrad(const size_t size, const T dout, T *mid_res, T *input_b, T *input_a,
                                       cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t CalMul(const bool broadcast_flag, const size_t shape_len, const size_t *lft_shape,
                                   const size_t lft_num, const size_t *rht_shape, const size_t rht_num,
                                   const size_t *out_shape, const size_t out_num, const T *x0, const T *x1, T *y,
                                   cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_EINSUM_IMPL_CUH_
