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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_REPEAT_ELEMENTS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_REPEAT_ELEMENTS_H_

#include <cuda_runtime.h>

#define REPEAT_ELEMENTS_MAX_INPUT_DIM 100

template <typename T>
void CalRepeatElements1d(
    const T *input, const int rep, const int axis, T *output, const int output_size, cudaStream_t cuda_stream);

template <typename T>
void CalRepeatElements2d(const T *input, const int input_d1, const int rep, const int axis, T *output,
                         const int output_d1, const int output_size, cudaStream_t cuda_stream);

template <typename T>
void CalRepeatElements3d(const T *input, const int input_d1, const int input_d2, const int rep, const int axis,
                         T *output, const int output_d1, const int output_d2, const int output_size,
                         cudaStream_t cuda_stream);

template <typename T>
void CalRepeatElements4d(const T *input, const int input_d1, const int input_d2, const int input_d3, const int rep,
                         const int axis, T *output, const int output_d1, const int output_d2, const int output_d3,
                         const int output_size, cudaStream_t cuda_stream);

template <typename T>
void CalRepeatElements5d(const T *input, const int input_d1, const int input_d2, const int input_d3, const int input_d4,
                         const int rep, const int axis, T *output, const int output_d1, const int output_d2,
                         const int output_d3, const int output_d4, const int output_size, cudaStream_t cuda_stream);

template <typename T>
void CalRepeatElements(const T *input, const int input_dim, const int* const input_shape,
                       const int* const input_shape_cumulative_product, const int rep, const int axis, T *output,
                       const int* const output_shape, const int output_size, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_REPEAT_ELEMENTS_H_
