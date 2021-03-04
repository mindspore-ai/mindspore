/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "transpose_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void Transpose(const size_t size, const T *input, const size_t *input_shape, const size_t *input_axis,
                          const size_t shape_size, T *output) {
  size_t pos_size;
  size_t temp_pos;
  size_t newpos;
  size_t newpos_size;
  size_t pos_array[TRANSPOSE_MAX_DIMENSION];

  // for example 4-D: pos = posArray[0] * input_shape[1] * input_shape[2] * input_shape[3] +
  //                        posArray[1] * input_shape[2] * input_shape[3] +
  //                        posArray[2] * input_shape[3] +
  //                        posArray[3]
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    temp_pos = pos;
    pos_size = size / input_shape[0];
    pos_array[0] = temp_pos / pos_size;
    for (size_t i = 1; i < shape_size; i++) {
      temp_pos -= pos_array[i - 1] * pos_size;
      pos_size = pos_size / input_shape[i];
      pos_array[i] = temp_pos / pos_size;
    }

    newpos = pos_array[input_axis[shape_size - 1]];
    newpos_size = 1;
    for (int64_t j = shape_size - 2; j >= 0; j--) {
      newpos_size *= input_shape[input_axis[j + 1]];
      newpos += pos_array[input_axis[j]] * newpos_size;
    }

    output[newpos] = input[pos];
  }
  return;
}
template <typename T>
void CalTranspose(const size_t size, const T *input, const size_t *input_shape, const size_t *input_axis,
                  const size_t shape_size, T *output, cudaStream_t cuda_stream) {
  Transpose<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, input_shape, input_axis, shape_size,
                                                               output);
  return;
}

template void CalTranspose<double>(const size_t size, const double *input, const size_t *input_shape,
                                   const size_t *input_axis, const size_t shape_size, double *output,
                                   cudaStream_t cuda_stream);
template void CalTranspose<float>(const size_t size, const float *input, const size_t *input_shape,
                                  const size_t *input_axis, const size_t shape_size, float *output,
                                  cudaStream_t cuda_stream);
template void CalTranspose<half>(const size_t size, const half *input, const size_t *input_shape,
                                 const size_t *input_axis, const size_t shape_size, half *output,
                                 cudaStream_t cuda_stream);
template void CalTranspose<int>(const size_t size, const int *input, const size_t *input_shape,
                                const size_t *input_axis, const size_t shape_size, int *output,
                                cudaStream_t cuda_stream);
template void CalTranspose<int64_t>(const size_t size, const int64_t *input, const size_t *input_shape,
                                    const size_t *input_axis, const size_t shape_size, int64_t *output,
                                    cudaStream_t cuda_stream);
