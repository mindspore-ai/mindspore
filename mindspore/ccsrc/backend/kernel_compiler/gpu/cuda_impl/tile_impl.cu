/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/tile_impl.cuh"

template <typename T>
__global__ void Tile(const size_t output_size, const size_t input_size, const size_t shape_size,
                     const size_t *input_shape, const size_t *output_shape, const T *input, T *output) {
  // for example 4-D: pos = pos_array[0] * output_shape[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[2] * output_shape[3] +
  //                        pos_array[3]
  size_t pos_array[TILE_MAX_DIMENSION];
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size; pos += blockDim.x * gridDim.x) {
    size_t tmp_pos = pos;
    size_t pos_size = output_size / output_shape[0];
    pos_array[0] = tmp_pos / pos_size;
    for (size_t i = 1; i < shape_size; i++) {
      tmp_pos -= pos_array[i - 1] * pos_size;
      pos_size = pos_size / output_shape[i];
      pos_array[i] = tmp_pos / pos_size;
    }
    for (size_t i = 0; i < shape_size; i++) {
      pos_array[i] = pos_array[i] % input_shape[i];
    }
    pos_size = input_size;
    size_t input_pos = 0;
    for (size_t i = 0; i < shape_size; i++) {
      pos_size /= input_shape[i];
      input_pos += (pos_array[i] * pos_size);
    }
    output[pos] = input[input_pos];
  }
}

template <typename T>
void CalTile(const size_t output_size, const size_t input_size, const size_t shape_size, const size_t *input_shape,
             const size_t *output_shape, const T *input, T *output, cudaStream_t cuda_stream) {
  Tile<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(output_size, input_size, shape_size, input_shape,
                                                                 output_shape, input, output);
  return;
}

template void CalTile<double>(const size_t output_size, const size_t input_size, const size_t shape_size,
                              const size_t *input_shape, const size_t *output_shape, const double *input,
                              double *output, cudaStream_t cuda_stream);
template void CalTile<float>(const size_t output_size, const size_t input_size, const size_t shape_size,
                             const size_t *input_shape, const size_t *output_shape, const float *input, float *output,
                             cudaStream_t cuda_stream);
template void CalTile<half>(const size_t output_size, const size_t input_size, const size_t shape_size,
                            const size_t *input_shape, const size_t *output_shape, const half *input, half *output,
                            cudaStream_t cuda_stream);
template void CalTile<int16_t>(const size_t output_size, const size_t input_size, const size_t shape_size,
                               const size_t *input_shape, const size_t *output_shape, const int16_t *input,
                               int16_t *output, cudaStream_t cuda_stream);
template void CalTile<int>(const size_t output_size, const size_t input_size, const size_t shape_size,
                           const size_t *input_shape, const size_t *output_shape, const int *input, int *output,
                           cudaStream_t cuda_stream);
template void CalTile<int64_t>(const size_t output_size, const size_t input_size, const size_t shape_size,
                               const size_t *input_shape, const size_t *output_shape, const int64_t *input,
                               int64_t *output, cudaStream_t cuda_stream);
template void CalTile<bool>(const size_t output_size, const size_t input_size, const size_t shape_size,
                               const size_t *input_shape, const size_t *output_shape, const bool *input,
                               bool *output, cudaStream_t cuda_stream);
