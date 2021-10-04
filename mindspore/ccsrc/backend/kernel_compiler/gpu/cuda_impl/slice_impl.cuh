/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SLICE_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SLICE_IMPL_CUH_

#include <cuda_runtime.h>
#include <vector>
#include "runtime/device/gpu/cuda_common.h"

template <typename T, typename...S>
void SliceKernel(const T *input, T *output, const size_t output_size, cudaStream_t cuda_stream, S...pack);

template <typename T>
void CalSlice4DGrad(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                    const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                    const size_t d3, const size_t d4, const T *dy, T *dx, cudaStream_t stream);

template <typename T>
void Slice1DKernel(const size_t s1, const size_t l1, const size_t d1, const T *input, T *output, cudaStream_t stream);

template <typename T>
void Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2, const size_t d1, const size_t d2,
                   const T *input, T *output, cudaStream_t stream);

template <typename T>
void Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1, const size_t l2, const size_t l3,
                   const size_t d1, const size_t d2, const size_t d3, const T *input, T *output, cudaStream_t stream);

template <typename T>
void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1, const size_t l2,
                   const size_t l3, const size_t l4, const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                   const T *input, T *output, cudaStream_t stream);

template <typename T>
void Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t l1,
                   const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                   const size_t d3, const size_t d4, const size_t d5, const T *input, T *output, cudaStream_t stream);

template <typename T>
void Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6,
                   const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                   const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                   const T *input, T *output, cudaStream_t stream);

template <typename T>
void Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6,
                   const size_t s7, const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                   const size_t l6, const size_t l7, const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                   const size_t d5, const size_t d6, const size_t d7, const T *input, T *output, cudaStream_t stream);

template <typename T>
void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                  const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape, const T *input,
                  T *output, cudaStream_t cuda_stream);

template <typename T>
void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                      const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape, const T *dy, T *dx,
                      cudaStream_t cuda_stream);

template <typename T>
void FillDeviceArray(const size_t input_size, T *addr, const float value, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SLICE_IMPL_CUH_
