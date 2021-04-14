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

#include <cuda_runtime.h>
#include "spacetodepth_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void SpaceToDepth(const size_t size, const T *input, const size_t *input_shape, const size_t *output_shape,
                             const size_t r, T *output) {
  size_t temp_stride = 0;
  size_t temp_pos = 0;
  size_t output_pos = 0;
  size_t input_pos_array[SPACETODEPTH_BUFFER_DIMENSION];
  size_t output_pos_array[SPACETODEPTH_BUFFER_DIMENSION];

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    temp_stride = input_shape[1] * input_shape[2] * input_shape[3];
    input_pos_array[0] = pos / temp_stride;
    temp_pos = pos % temp_stride;

    for (size_t i = 1; i < SPACETODEPTH_BUFFER_DIMENSION; i++) {
      temp_stride /= input_shape[i];
      input_pos_array[i] = temp_pos / temp_stride;
      temp_pos %= temp_stride;
    }

    output_pos_array[0] = input_pos_array[0];
    output_pos_array[1] = input_pos_array[1] * r * r + r * (input_pos_array[2] % r) + input_pos_array[3] % r;
    output_pos_array[2] = input_pos_array[2] / r;
    output_pos_array[3] = input_pos_array[3] / r;

    for (size_t i = 0; i < 3; ++i) {
      output_pos += output_pos_array[i];
      output_pos *= output_shape[i + 1];
    }
    output_pos += output_pos_array[3];

    output[output_pos] = input[pos];
  }
  return;
}

template <typename T>
void CalSpaceToDepth(const size_t size, const T *input, const size_t *input_shape, const size_t *output_shape,
                     const size_t r, T *output, cudaStream_t cuda_stream) {
  SpaceToDepth<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, input_shape, output_shape, r, output);
  return;
}

template void CalSpaceToDepth<float>(const size_t size, const float *input, const size_t *input_shape,
                                     const size_t *output_shape, const size_t r, float *output,
                                     cudaStream_t cuda_stream);
template void CalSpaceToDepth<half>(const size_t size, const half *input, const size_t *input_shape,
                                    const size_t *output_shape, const size_t r, half *output, cudaStream_t cuda_stream);
template void CalSpaceToDepth<int>(const size_t size, const int *input, const size_t *input_shape,
                                   const size_t *output_shape, const size_t r, int *output, cudaStream_t cuda_stream);
template void CalSpaceToDepth<int64_t>(const size_t size, const int64_t *input, const size_t *input_shape,
                                       const size_t *output_shape, const size_t r, int64_t *output,
                                       cudaStream_t cuda_stream);
template void CalSpaceToDepth<int16_t>(const size_t size, const int16_t *input, const size_t *input_shape,
                                       const size_t *output_shape, const size_t r, int16_t *output,
                                       cudaStream_t cuda_stream);
template void CalSpaceToDepth<int8_t>(const size_t size, const int8_t *input, const size_t *input_shape,
                                      const size_t *output_shape, const size_t r, int8_t *output,
                                      cudaStream_t cuda_stream);
template void CalSpaceToDepth<uint8_t>(const size_t size, const uint8_t *input, const size_t *input_shape,
                                       const size_t *output_shape, const size_t r, uint8_t *output,
                                       cudaStream_t cuda_stream);
template void CalSpaceToDepth<uint16_t>(const size_t size, const uint16_t *input, const size_t *input_shape,
                                        const size_t *output_shape, const size_t r, uint16_t *output,
                                        cudaStream_t cuda_stream);
template void CalSpaceToDepth<uint32_t>(const size_t size, const uint32_t *input, const size_t *input_shape,
                                        const size_t *output_shape, const size_t r, uint32_t *output,
                                        cudaStream_t cuda_stream);
template void CalSpaceToDepth<uint64_t>(const size_t size, const uint64_t *input, const size_t *input_shape,
                                        const size_t *output_shape, const size_t r, uint64_t *output,
                                        cudaStream_t cuda_stream);
