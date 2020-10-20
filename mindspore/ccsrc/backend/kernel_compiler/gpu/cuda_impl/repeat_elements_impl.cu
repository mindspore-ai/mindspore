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
#include <cuda_runtime.h>

#include "repeat_elements_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void RepeatElements1d(const T *input, const int rep, const int axis, T *output,
                                 const int output_size) {
  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < output_size; gt_id += gridDim.x * blockDim.x) {
    int copied_value_index = gt_id / rep;
    output[gt_id] = input[copied_value_index];
  }
}

template <typename T>
__global__ void RepeatElements2d(const T *input, const int input_d1, const int rep, const int axis, T *output,
                                 const int output_d1, const int output_size) {
  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < output_size; gt_id += gridDim.x * blockDim.x) {
    int global_array_index = gt_id;

    int index_d1 = global_array_index % output_d1;
    global_array_index -= index_d1;
    global_array_index /= output_d1;

    int index_d0 = global_array_index;

    switch (axis) {
      case 0:
        index_d0 /= rep;
        break;
      case 1:
        index_d1 /= rep;
        break;
    }

    const int term0 = index_d0 * input_d1;
    const int copied_value_index = term0 + index_d1;
    output[gt_id] = input[copied_value_index];
  }
}

template <typename T>
__global__ void RepeatElements3d(const T *input, const int input_d1, const int input_d2, const int rep, const int axis,
                                 T *output, const int output_d1, const int output_d2, const int output_size) {
  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < output_size; gt_id += gridDim.x * blockDim.x) {
    int global_array_index = gt_id;

    int index_d2 = global_array_index % output_d2;
    global_array_index -= index_d2;
    global_array_index /= output_d2;

    int index_d1 = global_array_index % output_d1;
    global_array_index -= index_d1;
    global_array_index /= output_d1;

    int index_d0 = global_array_index;

    switch (axis) {
      case 0:
        index_d0 /= rep;
        break;
      case 1:
        index_d1 /= rep;
        break;
      case 2:
        index_d2 /= rep;
        break;
      default:
        asm("trap;");
    }

    const int term0 = index_d0 * input_d1 * input_d2;
    const int term1 = index_d1 * input_d2;
    const int copied_value_index = term0 + term1 + index_d2;
    output[gt_id] = input[copied_value_index];
  }
}

template <typename T>
__global__ void RepeatElements4d(const T *input, const int input_d1, const int input_d2, const int input_d3,
                                 const int rep, const int axis, T *output, const int output_d1, const int output_d2,
                                 const int output_d3, const int output_size) {
  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < output_size; gt_id += gridDim.x * blockDim.x) {
    int global_array_index = gt_id;

    int index_d3 = global_array_index % output_d3;
    global_array_index -= index_d3;
    global_array_index /= output_d3;

    int index_d2 = global_array_index % output_d2;
    global_array_index -= index_d2;
    global_array_index /= output_d2;

    int index_d1 = global_array_index % output_d1;
    global_array_index -= index_d1;
    global_array_index /= output_d1;

    int index_d0 = global_array_index;

    switch (axis) {
      case 0:
        index_d0 /= rep;
        break;
      case 1:
        index_d1 /= rep;
        break;
      case 2:
        index_d2 /= rep;
        break;
      case 3:
        index_d3 /= rep;
        break;
    }

    const int term0 = index_d0 * input_d1 * input_d2 * input_d3;
    const int term1 = index_d1 * input_d2 * input_d3;
    const int term2 = index_d2 * input_d3;
    const int copied_value_index = term0 + term1 + term2 + index_d3;
    output[gt_id] = input[copied_value_index];
  }
}

template <typename T>
__global__ void RepeatElements5d(const T *input, const int input_d1, const int input_d2, const int input_d3,
                                 const int input_d4, const int rep, const int axis, T *output, const int output_d1,
                                 const int output_d2, const int output_d3, const int output_d4, const int output_size) {
  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < output_size; gt_id += gridDim.x * blockDim.x) {
    int global_array_index = gt_id;

    int index_d4 = global_array_index % output_d4;
    global_array_index -= index_d4;
    global_array_index /= output_d4;

    int index_d3 = global_array_index % output_d3;
    global_array_index -= index_d3;
    global_array_index /= output_d3;

    int index_d2 = global_array_index % output_d2;
    global_array_index -= index_d2;
    global_array_index /= output_d2;

    int index_d1 = global_array_index % output_d1;
    global_array_index -= index_d1;
    global_array_index /= output_d1;

    int index_d0 = global_array_index;

    switch (axis) {
      case 0:
        index_d0 /= rep;
        break;
      case 1:
        index_d1 /= rep;
        break;
      case 2:
        index_d2 /= rep;
        break;
      case 3:
        index_d3 /= rep;
        break;
      case 4:
        index_d4 /= rep;
        break;
    }

    const int term0 = index_d0 * input_d1 * input_d2 * input_d3 * input_d4;
    const int term1 = index_d1 * input_d2 * input_d3 * input_d4;
    const int term2 = index_d2 * input_d3 * input_d4;
    const int term3 = index_d3 * input_d4;
    const int copied_value_index = term0 + term1 + term2 + term3 + index_d4;
    output[gt_id] = input[copied_value_index];
  }
}

template <typename T>
__global__ void RepeatElements(const T *input, const int input_dim, const int* const input_shape,
                               const int* const coefficients, const int rep, const int axis, T *output,
                               const int* const output_shape, const int output_size) {
  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < output_size; gt_id += gridDim.x * blockDim.x) {
    int index_tuple[REPEAT_ELEMENTS_MAX_INPUT_DIM];

    int global_array_index = gt_id;
    for (size_t i = input_dim - 1; i > 0; i--) {
      int coordinate = global_array_index % output_shape[i];
      index_tuple[i] = coordinate;
      global_array_index -= coordinate;
      global_array_index /= output_shape[i];
    }
    index_tuple[0] = global_array_index;

    index_tuple[axis] /= rep;

    int copied_value_index = 0;
    for (size_t i = 0; i < input_dim - 1; i++) {
      copied_value_index += index_tuple[i] * coefficients[i];
    }
    copied_value_index += index_tuple[input_dim - 1];

    output[gt_id] = input[copied_value_index];
  }
}

template <typename T>
void CalRepeatElements1d(
    const T *input, const int rep, const int axis, T *output, const int output_size, cudaStream_t cuda_stream) {
  RepeatElements1d<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(input, rep, axis, output, output_size);
}

template <typename T>
void CalRepeatElements2d(const T *input, const int input_d1, const int rep, const int axis, T *output,
                         const int output_d1, const int output_size, cudaStream_t cuda_stream) {
  RepeatElements2d<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(input, input_d1, rep, axis, output,
                                                                             output_d1, output_size);
}

template <typename T>
void CalRepeatElements3d(const T *input, const int input_d1, const int input_d2, const int rep, const int axis,
                         T *output, const int output_d1, const int output_d2, const int output_size,
                         cudaStream_t cuda_stream) {
  RepeatElements3d<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(input, input_d1, input_d2, rep, axis,
                                                                             output, output_d1, output_d2, output_size);
}

template <typename T>
void CalRepeatElements4d(const T *input, const int input_d1, const int input_d2, const int input_d3, const int rep,
                         const int axis, T *output, const int output_d1, const int output_d2, const int output_d3,
                         const int output_size, cudaStream_t cuda_stream) {
  RepeatElements4d<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(input, input_d1, input_d2, input_d3, rep,
                                                                             axis, output, output_d1, output_d2,
                                                                             output_d3, output_size);
}

template <typename T>
void CalRepeatElements5d(const T *input, const int input_d1, const int input_d2, const int input_d3, const int input_d4,
                         const int rep, const int axis, T *output, const int output_d1, const int output_d2,
                         const int output_d3, const int output_d4, const int output_size, cudaStream_t cuda_stream) {
  RepeatElements5d<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(input, input_d1, input_d2, input_d3,
                                                                             input_d4, rep, axis, output, output_d1,
                                                                             output_d2, output_d3, output_d4,
                                                                             output_size);
}

template <typename T>
void CalRepeatElements(const T *input, const int input_dim, const int* const input_shape,
                       const int* const input_shape_cumulative_product, const int rep, const int axis, T *output,
                       const int* const output_shape, const int output_size, cudaStream_t cuda_stream) {
  RepeatElements<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(input, input_dim, input_shape,
                                                                           input_shape_cumulative_product, rep, axis,
                                                                           output, output_shape, output_size);
}

// int32
template void CalRepeatElements1d<int>(
    const int *input, const int rep, const int axis, int *output, const int output_size, cudaStream_t cuda_stream);

template void CalRepeatElements2d<int>(const int *input, const int input_d1, const int rep, const int axis, int *output,
                                       const int output_d1, const int output_size, cudaStream_t cuda_stream);

template void CalRepeatElements3d<int>(const int *input, const int input_d1, const int input_d2, const int rep,
                                       const int axis, int *output, const int output_d1, const int output_d2,
                                       const int output_size, cudaStream_t cuda_stream);

template void CalRepeatElements4d<int>(const int *input, const int input_d1, const int input_d2, const int input_d3,
                                       const int rep, const int axis, int *output, const int output_d1,
                                       const int output_d2, const int output_d3, const int output_size,
                                       cudaStream_t cuda_stream);

template void CalRepeatElements5d<int>(const int *input, const int input_d1, const int input_d2, const int input_d3,
                                       const int input_d4, const int rep, const int axis, int *output,
                                       const int output_d1, const int output_d2, const int output_d3,
                                       const int output_d4, const int output_size, cudaStream_t cuda_stream);

template void CalRepeatElements<int>(const int *input, const int input_dim, const int* const input_shape,
                                     const int* const input_shape_cumulative_product, const int rep, const int axis,
                                     int *output, const int* const output_shape, const int output_size,
                                     cudaStream_t cuda_stream);

// float16
template void CalRepeatElements1d<half>(
    const half *input, const int rep, const int axis, half *output, const int output_size, cudaStream_t cuda_stream);

template void CalRepeatElements2d<half>(const half *input, const int input_d1, const int rep, const int axis,
                                        half *output, const int output_d1, const int output_size,
                                        cudaStream_t cuda_stream);

template void CalRepeatElements3d<half>(const half *input, const int input_d1, const int input_d2, const int rep,
                                        const int axis, half *output, const int output_d1, const int output_d2,
                                        const int output_size, cudaStream_t cuda_stream);

template void CalRepeatElements4d<half>(const half *input, const int input_d1, const int input_d2, const int input_d3,
                                        const int rep, const int axis, half *output, const int output_d1,
                                        const int output_d2, const int output_d3, const int output_size,
                                        cudaStream_t cuda_stream);

template void CalRepeatElements5d<half>(const half *input, const int input_d1, const int input_d2, const int input_d3,
                                        const int input_d4, const int rep, const int axis, half *output,
                                        const int output_d1, const int output_d2, const int output_d3,
                                        const int output_d4, const int output_size, cudaStream_t cuda_stream);

template void CalRepeatElements<half>(const half *input, const int input_dim, const int* const input_shape,
                                      const int* const input_shape_cumulative_product, const int rep, const int axis,
                                      half *output, const int* const output_shape, const int output_size,
                                      cudaStream_t cuda_stream);
